import json
import logging
import posixpath as pp
import re
from typing import Tuple

import fsspec
import numpy as np
import pandas as pd

from .model_api import GPTModelCall
from .post_processing import (
    _complete_data_dict,
    _detect_alter_drop_table,
    _extract_queries,
    _string_to_dict,
    _update_percent_sign,
    extract_code,
    round_float_numbers_in_dataframe,
)
from .utils import rate_limit_error_handler, read_text_file

# import sys


# from .utils import MyLogger

MYLOGGERNAME = "QueryInsights"


class TextToQuery:
    """
    Track 1 - to generate SQL query from the user question

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    debug_config: dict
        Debug config dictionary for using appropriate prompts to make requests for debugging to GPT.
    question : str
        User question to be answered
    additional_context : str
        Additional context to answer the question
    data_dictionary : dict
        contains table name, column name and description
    db_connection :
        to connect to SQL DB
    output_path : str
        path to save the results
    skip_model : bool
        condition whether to skip the api call.
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    similarity: list
        [Boolean, (Question index, Similarity score, Question)]
        Boolean represents whether the user question is similar to any existing question.
        If First value is True, Second element (Tuple) will be populated. Otherwise it will have None values.
        Question index is the index of the similar question.
        Similarity score represents how similar the user question is to the existing question.
    """

    def __init__(
        self,
        user_config,
        model_config,
        debug_config,
        question,
        additional_context,
        data_dictionary,
        business_overview,
        bot_history,
        db_connection,
        output_path,
        similarity,
        skip_model: bool = False,
        fs=None,
    ) -> None:
        """Class constructor"""
        # Config related
        self.prompt_dict = model_config.text_to_query.prompts
        self.model_param_dict = model_config.text_to_query.model_params
        self.model_param_dict["history"] = bot_history

        self.connection_param_dict = user_config.connection_params
        self.text_to_query_debug_dict = debug_config.text_to_query
        self.ui = user_config.ui

        # Business user query related
        self.question = question
        self.additional_context = additional_context
        self.business_overview = business_overview
        self.raw_data_dictionary = data_dictionary
        self.conn = db_connection
        self.output_path = output_path
        self.skip_model = skip_model
        self.similarity = similarity

        # Logger
        self.logger = logging.getLogger(MYLOGGERNAME)

        # Init some instance vars to None
        self.output_table = None
        self.output_query = None
        self.output_table_dict = None

        self._fs = fs or fsspec.filesystem("file")
        self.prompt_dict["static_prompt_original"] = self.prompt_dict["static_prompt"]

        # Check if there is a similar question identified.
        if self.similarity[0]:
            # Create a filtered data dictionary based on the columns identified using the similar query.
            self.data_dictionary = {
                table_name: {
                    "table_name": table_data["table_name"],
                    "columns": [
                        {key: value for key, value in column.items() if key != "similar"}
                        for column in table_data["columns"]
                        if column.get("id") == "Yes" or column.get("similar") == "Yes"
                    ],
                }
                for table_name, table_data in self.raw_data_dictionary.items()
                if any(column.get("similar") == "Yes" for column in table_data["columns"])
            }
            # Update the prompt for similar questions and assign the similar question/response as the sample input
            self.prompt_dict["static_prompt"] = self.prompt_dict["static_prompt_similar_question"]
            sample_input = self.similarity[1]
        else:
            # Use the original data dictionary
            self.data_dictionary = self.raw_data_dictionary
            # No sample input since similarity is False
            sample_input = None

        # GPT Model Call object
        self.text_to_query_ins = GPTModelCall(
            prompt_dict=self.prompt_dict,
            question=self.question,
            additional_context=self.additional_context,
            connection_param_dict=self.connection_param_dict,
            dictionary=self.data_dictionary,
            business_overview=self.business_overview,
            sample_input=sample_input,
        )

        # Save prompts if we are going to run track 1.
        if not self.skip_model:
            self.logger.debug("Saving prompts.")
            with self._fs.open(pp.join(self.output_path, "prompt.txt"), "w") as f:
                f.writelines(self.text_to_query_ins.prompt)

        # Required for decorator
        time_delay = user_config.time_delay
        max_retries = model_config.text_to_query.model_params.max_tries

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._call_model_api = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries
        )(self._call_model_api)
        return

    def _call_model_api(self, with_history: bool = False, history_kwargs: dict = None):
        """
        call_model_api
        Get model response from GPT model
        """
        if not with_history:
            self.logger.debug("Sending a new request to GPT")
            (
                query_model_output,
                query_model_finish,
                query_model_tokens,
            ) = self.text_to_query_ins.model_response(self.model_param_dict)
        else:
            self.logger.debug("Requesting GPT to debug generated SQL.")
            (
                query_model_output,
                query_model_finish,
                query_model_tokens,
            ) = self.text_to_query_ins.model_response(**history_kwargs)

        if (not self.skip_model) and hasattr(self.text_to_query_ins, "current_message"):
            self.logger.debug("Saving the current_message input to GPT.")
            with self._fs.open(
                pp.join(self.output_path, "current_message_input.json"), "w"
            ) as f:
                json.dump(self.text_to_query_ins.current_message, f, indent=4)

        self.logger.info(
            f"Track 1:-\n finish token - {query_model_finish},\n token information - {query_model_tokens}"
        )
        self.logger.debug(f"Model output\n{query_model_output}")

        return query_model_output, query_model_finish, query_model_tokens

    def _clean_gpt_response(
        self, raw_response: str, start_pattern: list = ["<start>"], end_pattern: str = "<end>"
    ) -> str:
        """Cleans GPT response by trimming prefixes and suffixes

        Parameters
        ----------
        raw_response : str
            GPT response
        start_pattern : list, optional
            prefix to be removed, by default ["<start>"]
        end_pattern : str, optional
            suffix to be removed, by default "<end>"

        Returns
        -------
        str
            trimmed response
        """
        cleaned_response = extract_code(
            string_input=raw_response, start=start_pattern, end=end_pattern, extract="first"
        )
        if cleaned_response is None:
            if end_pattern == "<end_dict>":
                cleaned_response = ""
            else:
                cleaned_response = raw_response

        return cleaned_response

    def _clean_sql(self, raw_sql: str) -> str:
        """Post processing of GPT generated SQL

        Parameters
        ----------
        raw_sql : str
            SQL that needs to be cleaned

        Returns
        -------
        str
            cleaned SQL

        Raises
        ------
        pd.io.sql.DatabaseError
            when we encounter ALTER/CREATE SQL we raise this error
        """
        # Look for multiple SQL queries and check if all are starting with Select or with
        self.logger.debug("Checking for multiple SQL query.")
        self.queries_list = _extract_queries(raw_sql)
        self.logger.info(f"Number of SQL Queries present - {len(self.queries_list)}")

        # for i in range(0, len(self.queries_list)):
        #     self.logger.info(f"{self.queries_list[i]}")
        # self.logger.info(f"Latest Query - {self.queries_list[-1][0]}")

        # TODO: Look for alternate ways the SQL comments can come and handle in the post processing.
        # Once that is handled, the detect_alter_drop_table is redundant. Then the logic can be changed to
        # run the latest SELECT statement, even though ALTER/UPDATE statements are present.
        if _detect_alter_drop_table(raw_sql):
            error_msg = f"Execution failed on sql '{raw_sql}': ALTER or DROP TABLE query detected."
            self.logger.error(f"{error_msg}")

            # Assign this var in case of alter table
            self.alter_query = raw_sql

            raise pd.io.sql.DatabaseError(error_msg)
        # Check if all SQL Queries present are starting with SELECT or WITH
        elif len(self.queries_list) > 1 and all(item[1] for item in self.queries_list):
            self.logger.info(
                "Multiple select statements found. Extracting the latest Select query."
            )
            cleaned_query = self.queries_list[-1][0]
        else:
            self.logger.info(
                "Single SQL query present. No ALTER/UPDATE/DROP/CREATE statements found."
            )
            cleaned_query = raw_sql

        # Removing percent sign from the final executable query and executing the query.
        cleaned_query = _update_percent_sign(cleaned_query, update="remove")

        return cleaned_query

    def _post_process_response(self, raw_response: str) -> str:
        """Post processing that cleans the response to extract SQL and post process the SQL to be ready to run.

        Parameters
        ----------
        raw_response : str
            GPT response

        Returns
        -------
        str
            cleaned SQL and data dictionary
        """
        # Extract query and data dict from response
        output_query = self._clean_gpt_response(
            raw_response=raw_response, start_pattern=["<start>"], end_pattern="<end>"
        )
        # sometimes, response will also contain ``` instead of tags.
        if "```" in output_query:
            output_query = self._clean_gpt_response(
                raw_response=raw_response, start_pattern=["```"], end_pattern="```"
            )

        output_table_dict = self._clean_gpt_response(
            raw_response=raw_response,
            start_pattern=["<start_dict>"],
            end_pattern="<end_dict>",
        )
        self.logger.info("SQL and data dict extracted.")

        # save model response and SQL query before running so that we can use it for debugging if something errors out in the middle.
        self.logger.debug("Saving query.")
        with self._fs.open(pp.join(self.output_path, "sql_query.sql"), "w") as f:
            f.writelines(output_query)

        self.logger.debug("Saving model response.")
        with self._fs.open(pp.join(self.output_path, "model_response.txt"), "w") as f:
            f.writelines(raw_response)

        self.logger.debug("Cleaning SQL")
        final_query = self._clean_sql(raw_sql=output_query)

        return final_query, output_table_dict

    def _post_process_output(self) -> None:
        """Post proess the output of GPT after running the SQL

        Raises
        ------
        ValueError
            if any of the column has invalid format.
        """
        # Replace string 'None' witn nan
        self.output_table = self.output_table.replace("None", np.nan)
        self.logger.info("Results from SQL is extracted.")

        if self.output_table.empty:
            error_msg = "No data is fetched while running the SQL.\nPlease change the user question or the data."
            self.logger.error(error_msg + "query:\n" + self.output_query)
            raise ValueError(error_msg)

        # Post process data dictionary
        self.output_table_dict = _complete_data_dict(
            output_table=self.output_table,
            raw_dict=self.data_dictionary,
            result_dict=_string_to_dict(self.output_table_dict),
        )

        # Round float numbers to consistent format
        self.output_table = round_float_numbers_in_dataframe(self.output_table)

        # Identify numeric and datetime columns
        num_cols = []
        date_cols = []
        for col in self.output_table.columns:
            try:
                self.output_table[col] = pd.to_numeric(self.output_table[col])
                num_cols.append(col)
            except ValueError:
                # Identify datetime columns
                try:
                    self.output_table[col] = pd.to_datetime(self.output_table[col])
                    date_cols.append(col)
                except ValueError:
                    pass

        return

    def _extract_sql_and_error(self, error_msg: str) -> Tuple[str, str]:
        """Based on error pattern, extracts the failed SQL and column name/error description

        Parameters
        ----------
        error_msg : str
            Error message for which we are extracting

        Returns
        -------
        Tuple[str, str]
            Error SQL and column name/error description
        """
        # Extract column that is invalid
        error_parts = error_msg.split(":")
        error_parts = [part.strip() for part in error_parts]

        # First part is SQL, capture everything that is between two single quotes.
        error_sql = re.findall(pattern=r"'([^']*)'", string=error_parts[0])

        # Second part is actual error message in case of 'no such column'
        # Final part is invalid column name with alias or actual error message for 'syntax error'
        final_part = error_parts[-1].split(".")[-1]

        return error_sql, final_part

    def _construct_debug_prompt_sql(self, debug_handler_dict: dict, error_msg: str) -> str:
        """Constructs debug prompt for SQL.

        Parameters
        ----------
        debug_handler_dict : dict
            debug dictionary
        error_msg : str
            Actual error message for which we are debugging

        Returns
        -------
        str
            newly constructed debug prompt.
        """
        # Identify the error based on history
        error_under_consideration = None
        for error_type, error_params in debug_handler_dict.items():
            if error_params["error_pattern"] in error_msg:
                error_under_consideration = error_type
                break

        # error_under_consideration = "wrongerror"  # Test
        # Track 1 errors
        if error_under_consideration in ["error_invalid_column", "error_ambiguous_column"]:
            _, error_column_name = self._extract_sql_and_error(error_msg=error_msg)
            debug_prompt = debug_handler_dict[error_under_consideration]["debug_prompt"].replace(
                "<column_name>", error_column_name
            )
        elif error_under_consideration in ["error_syntax", "error_alter"]:
            _, _ = self._extract_sql_and_error(error_msg=error_msg)
            debug_prompt = debug_handler_dict[error_under_consideration]["debug_prompt"]
        else:
            raise ValueError(
                f"Below SQL debugging is not handled.\n\nSQL error description:\n{error_msg}"
            )

        return debug_prompt

    def _run_sql(self):
        """
        post_processing
        Process the model result to extract the SQL query and the data dictionary to use them the following steps. Also fix data type issues in the code
        """
        try:
            # SQL execution block
            self.output_query, self.output_table_dict = self._post_process_response(
                raw_response=self.query_model_output
            )
            self.logger.info("Running SQL.")
            self.output_table = pd.read_sql_query(self.output_query, self.conn)

            # Check if the table has 0 rows. If so, add % to the LIKE if any and re-execute the query.
            if self.output_table.shape[0] == 0 and "LIKE" in self.output_query:
                self.logger.info(
                    "The table returned has 0 rows. Trying another iteration by adding % to the LIKE paramater if any."
                )
                self.output_query = _update_percent_sign(self.output_query, update="add")
                self.output_table = pd.read_sql_query(self.output_query, self.conn)

        except pd.io.sql.DatabaseError as err_msg:
            # DEBUG ERROR SQL Section
            err_msg = str(err_msg)  # Convert to string from error object.
            self.logger.error(f"Error raised when running SQL. Error description: {err_msg}")
            self.logger.info("Asking GPT to debug the SQL.")

            debug_prompt = self._construct_debug_prompt_sql(
                debug_handler_dict=self.text_to_query_debug_dict,
                error_msg=err_msg,
            )

            # Save prompts if we are going to run track 1.
            if not self.skip_model:
                self.logger.debug(
                    "Saving Modified prompts and moving older prompts to bin folder."
                )
                if self._fs.exists(pp.join(self.output_path, "prompt.txt")):
                    self._fs.makedirs(pp.join(self.output_path, "bin"), exist_ok=True)
                    self._fs.move(
                        pp.join(self.output_path, "prompt.txt"),
                        pp.join(self.output_path, "bin", "prompt.txt"),
                    )

                # Move old files to bin
                if self._fs.exists(pp.join(self.output_path, "sql_query.sql")):
                    self._fs.move(
                        pp.join(self.output_path, "sql_query.sql"),
                        pp.join(self.output_path, "bin", "sql_query.sql"),
                    )

                if self._fs.exists(pp.join(self.output_path, "model_response.txt")):
                    self._fs.move(
                        pp.join(self.output_path, "model_response.txt"),
                        pp.join(self.output_path, "bin", "model_response.txt"),
                    )

            if hasattr(self, "alter_query"):  # Alter query detected
                history_kwargs = {
                    "model_param_dict": self.model_param_dict,
                    "debug_prompt": debug_prompt,
                    "history": "Received error while running this SQL:\n" + self.alter_query,
                }
                new_prompt = (
                    self.text_to_query_ins.prompt
                    + "\n\n"
                    + "Received error while running this SQL:\n"
                    + self.alter_query
                    + "\n\n"
                    + debug_prompt
                )
            else:  # column not found and syntax error case.
                history_kwargs = {
                    "model_param_dict": self.model_param_dict,
                    "debug_prompt": debug_prompt,
                    "history": "Received error while running this SQL:\n" + self.output_query,
                }
                new_prompt = (
                    self.text_to_query_ins.prompt
                    + "\n\n"
                    + "Received error while running this SQL:\n"
                    + self.output_query
                    + "\n\n"
                    + debug_prompt
                )

            with self._fs.open(pp.join(self.output_path, "prompt.txt"), "w") as f:
                f.writelines(new_prompt)

            if (not self.skip_model)and (hasattr(self.text_to_query_ins, "current_message")):
                self.logger.debug("Saving the current_message input to GPT.")
                with self._fs.open(
                    pp.join(self.output_path, "current_message_input.json"), "w"
                ) as f:
                    json.dump(self.text_to_query_ins.current_message, f, indent=4)

            (
                self.query_model_output,
                self.query_model_finish,
                self.query_model_tokens,
            ) = self._call_model_api(with_history=True, history_kwargs=history_kwargs)

            self.logger.info("Received modified SQL after GPT debugging.")
            self.output_query, self.output_table_dict = self._post_process_response(
                raw_response=self.query_model_output
            )

            self.logger.debug("Saving modified query.")
            with self._fs.open(pp.join(self.output_path, "sql_query.sql"), "w") as f:
                f.writelines(self.output_query)

            self.logger.debug("Saving modified model response.")
            with self._fs.open(pp.join(self.output_path, "model_response.txt"), "w") as f:
                f.writelines(self.query_model_output)

            self.logger.info("Running modified SQL.")
            self.output_table = pd.read_sql_query(self.output_query, self.conn)

            # Check if the table has 0 rows. If so, add % to the LIKE if any and re-execute the query.
            if self.output_table.shape[0] == 0 and "LIKE" in self.output_query:
                self.logger.info(
                    "The table returned has 0 rows. Trying another iteration by adding % to the LIKE paramater if any."
                )
                self.output_query = _update_percent_sign(self.output_query, update="add")
                self.output_table = pd.read_sql_query(self.output_query, self.conn)
        except Exception as e:
            error_msg = f"""Error description - {e}
            Error occurred while running the GPT generated SQL for procuring the data.\nPlease change the user question or the data.
            """
            self.logger.error(error_msg + "query:\n" + self.output_query)

            raise ValueError(error_msg)

        self._post_process_output()
        return

    def _save_outputs(self):
        """
        save_outputs
        Save the outputs in the respective folders
        """
        # save output table
        self.logger.debug("Saving table.")
        with self._fs.open(
            pp.join(self.output_path, "output_table.csv"), mode="wb", newline=""
        ) as fp:
            self.output_table.to_csv(fp, index=False)

        # if not self.skip_model:
        #     self.logger.debug("Saving the current_message input to GPT.")
        #     with self._fs.open(pp.join(self.output_path, "current_message_input.json"), "w") as f:
        #         json.dump(self.text_to_query_ins.current_message, f, indent=4)

        # save output table data dictionary
        self.logger.debug("Saving data dictionary.")
        with self._fs.open(pp.join(self.output_path, "output_data_dictionary.txt"), "w") as f:
            f.writelines(f"{self.output_table_dict}")

        return

    def get_query_suggestion(self):
        """
        This is the main function call for SQL query and table generation.

        Flow -
            1. If it's the same question from KB, model response is retrieved from KB folders.
                - If it results in an error, normal API call happens.
            2. If the user question is similar to existing question in KB, model call happens with modified prompt.
                - If it results in an error, normal API call happens.
            3. In case of a new unrelated question, normal API call happens.

        Returns:
            None
        """
        if self.skip_model:
            try:
                self.logger.info("Question already exists. Getting SQL from knowledge base.")
                model_response = pp.join(self.output_path, "model_response.txt")
                self.query_model_output = read_text_file(model_response, fs=self._fs)
                # print(self.query_model_output)
                self.logger.info("SQL Query retreived from Knowledge Base.")
                self.process_query(skip_api_call=True)
            except Exception as e:
                self.logger.info(
                    f"Knowledge Retrieval failed with error {e}. Making the GPT call again."
                )
                self.process_query()
        elif self.similarity[0]:
            try:
                self.logger.info("Running GPT request for SQL generation for similar question.")
                self.process_query()
            except Exception as e:
                self.logger.info(f"Similar question api call ended with error - {e}.")
                self.logger.info("Running the usual GPT request for SQL generation.")
                # Updaing the static prompt to the usual one.
                self.prompt_dict["static_prompt"] = self.prompt_dict["static_prompt_original"]
                # Passing the raw data dictionary without sample input.
                self.text_to_query_ins = GPTModelCall(
                    prompt_dict=self.prompt_dict,
                    question=self.question,
                    additional_context=self.additional_context,
                    connection_param_dict=self.connection_param_dict,
                    dictionary=self.raw_data_dictionary,
                    business_overview=self.business_overview,
                    sample_input=None,
                )
                self.process_query()
        else:  # Normal run
            self.process_query()

        return

    def process_query(self, skip_api_call: bool = False):
        """Generates SQL from GPT and executes it. Returns the output table and its data dictionary if SQL execution yielded results.
        GPT call call can be skipped using the skip_api_call parameter if self.query_model_output is obtained from KB (applicable for same questions in KB).

        Parameters
        ----------
        skip_api_call : boolean (Default - False)
            Skips the GPT api call if this parameter is True.

        Returns:
            None
        """
        if not skip_api_call:
            self.logger.info("Calling the API for SQL generation.")
            (
                self.query_model_output,
                self.query_model_finish,
                self.query_model_tokens,
            ) = self._call_model_api()

        self.logger.info("Executing the SQL to generate output.")
        self._run_sql()

        self.logger.info("Saving the output to predefined paths.")
        self._save_outputs()

        self.logger.info("Text to Query is completed.")
        return


class BotResponse:
    def __init__(
        self,
        user_config=None,
        model_config=None,
        conversation_history=None,
        error_message=None,
        skip_model=False,
        mode="rule_based",
    ):
        if mode == "model_based":
            self.prompt_dict = model_config.bot_response.prompts
            self.model_param_dict = model_config.bot_response.model_params
            self.connection_param_dict = user_config.connection_params
            self.ui = user_config.ui
            self.conversation_history = ""
            for conv in conversation_history[:-1]:
                self.conversation_history += f"user: {conv[0]}\n"
                self.conversation_history += f"bot: {conv[1]}\n"
            self.conversation_history += f"user: {conversation_history[-1][0]}"
            self.initialize_model_attr()

            # Required for decorator
            time_delay = user_config.time_delay
            max_retries = model_config.text_to_query.model_params.max_tries

            # Normal way of using decorator as we are getting trouble passing arguments
            # in intended way of "@rate_limit_error_handler(...)"
            self._call_model_api = rate_limit_error_handler(
                logger=self.logger, time_delay=time_delay, max_retries=max_retries
            )(self._call_model_api)

        self.error_message = error_message

        self.skip_model = skip_model

        self.logger = logging.getLogger(MYLOGGERNAME)

        return

    def _call_model_api(self):
        """
        call_model_api
        Get model response from GPT model
        """
        self.logger.debug("call model api reached for bot response")
        self.bot_response_ins = GPTModelCall(
            prompt_dict=self.prompt_dict,
            question=None,
            additional_context=None,
            connection_param_dict=self.connection_param_dict,
            dictionary=None,
            business_overview=None,
            suggestion=None,
            table=None,
            history=self.conversation_history,
            error_message=self.error_message,
        )
        (
            self.bot_response_output,
            self.bot_response_finish,
            self.bot_response_tokens,
        ) = self.bot_response_ins.model_response(self.model_param_dict)
        self.bot_response_output = self.bot_response_output.replace("\n", "")
        self.logger.info(
            f"Track 1:-\n finish token - {self.bot_response_finish},\n token information - {self.bot_response_tokens}"
        )
        self.logger.debug(f"Model output\n{self.bot_response_output}")

    def process_sql_error(self):
        """calls the API to generate appropriate bot response when the generated SQL query fails.

        Returns:
            None
        """

        if self.skip_model:
            try:
                self.logger.info(
                    "Question already exists. Getting bot response from from knowledge base."
                )
                # TODO: handle skip model
                # model_response = os.path.join(self.output_path, "model_response.txt")
                # self.query_model_output = read_text_file(model_response)
                # print(self.query_model_output)
                # self.logger.info("SQL Query retreived from Knowledge Base.")
            except Exception as e:
                self.skip_model = False
                self.logger.info(
                    f"Knowledge Retrieval failed with error {e}. Making the GPT call again."
                )
                self._call_model_api()
        else:
            self.logger.info("Making the GPT request for bot response generation.")
            self._call_model_api()

        # self.logger.info("Saving the output to predefined paths.")
        # self._save_outputs()

        self.logger.info("Bot response generation is completed.")
        return

    def extract_key_error_from_message(self, error_message, key_phrase):
        """
        To extract the column/table/funtion name that caused the SQL error

        Parameters
        ----------
        error_message : str
            contains the error message genrated by pandas read_sql_query
        key_phrase : str
            contains the phrase (like: no such column, ambiguous column name) that can be used for splitting the string

        Returns
        -------
        str
            column/table/funtion name that caused the error
        """
        return error_message.split(key_phrase)[1].strip().split(" ")[0].strip().split(".")[-1]

    def get_bot_error_message(self, error_message):
        """
        Generates the bot response without any model using some custom messages for each error.

        Parameters
        ----------
        error_message : str
            contains the error message genrated by pandas read_sql_query

        Returns
        -------
        str
            custom bot response specific to the given error
        """
        error_message = error_message.replace("\n", " ")

        # OperationalError: no such column: <column_name>: This error occurs when the specified column does not exist in the table.
        if ": no such column:" in error_message:
            error = self.extract_key_error_from_message(error_message, ": no such column:")
            return f"I didn't get what you mean by this column: '{error}'. Help me in understand where to get it."

        # OperationalError: no such table: <table_name>: This error indicates that the specified table does not exist in the database.
        elif ": no such table:" in error_message:
            error = self.extract_key_error_from_message(error_message, ": no such table:")
            return f"I couldn't find the table: '{error}'. Help me understand which table to use."

        # OperationalError: near "<syntax_error>": syntax error: This error indicates a syntax error in the SQL query, where <syntax_error> represents the specific syntax element that caused the issue.
        elif ": syntax error:" in error_message:
            return "Looks like there is a syntax error. Please check the below query and help me correct it."

        # OperationalError: ambiguous column name: <column_name>: This error occurs when the specified column name is ambiguous and exists in multiple tables referenced in the query. To resolve this, you can specify the table name or alias along with the column name in the query to disambiguate it.
        elif ": ambiguous column name:" in error_message:
            error = self.extract_key_error_from_message(error_message, ": ambiguous column name:")
            return f"The column: '{error}' is present in more than one table. Help me identify which one to use."

        # OperationalError: no such function: <function_name>: This error occurs when the query references a function that does not exist in SQLite. Make sure that the function name is spelled correctly and that the function is supported by SQLite.
        elif ": no such function:" in error_message:
            error = self.extract_key_error_from_message(error_message, ": no such function:")
            return f"The fuction: '{error}' is not found. Can you ask me something else?"

        # OperationalError: unrecognized token: <token>: This error indicates that the query contains an unrecognized or invalid token. Check the query syntax and ensure that all keywords, operators, and identifiers are correctly specified.
        elif ": unrecognized token:" in error_message:
            error = self.extract_key_error_from_message(error_message, ": unrecognized token:")
            return f"The token: '{error}' is invalid. Check the query syntax and try rephrasing the question."

        # OperationalError: No data is fetched while running the SQL query
        elif "No data is fetched" in error_message:
            return "No data fetched. Please check the filters in the SQL query below and suggest any changes (if required)."

        else:
            return None
        # OperationalError: table <table_name> has no column named <column_name>: This error message indicates that the specified table does not have a column with the given name. Double-check the column name for typos or verify the table schema to ensure the column exists.
        # OperationalError: too many terms in compound SELECT: This error message suggests that the compound SELECT statement has an excessive number of terms, exceeding the limits set by SQLite. Review and simplify the compound SELECT statement to resolve this error.
        # OperationalError: database is locked: This error occurs when multiple processes or threads are attempting to access the database simultaneously and one of them has a lock on the database. It indicates a concurrency issue and typically resolves when the lock is released.
        # OperationalError: no such module: <module_name>: This error message suggests that the specified SQLite module is not available or not installed. SQLite allows the use of external modules, and this error occurs when trying to access a module that is not present.
        # OperationalError: disk I/O error: This error indicates a problem with the disk I/O operations, such as reading or writing to the database file. It can occur due to disk failures, lack of disk space, or file permission issues.
        # ProgrammingError: Incorrect number of bindings supplied: This error occurs when the number of provided parameter bindings in the SQL query does not match the number of placeholders. Ensure that the number of bindings matches the number of placeholders in the query.
        # ProgrammingError: Incorrect type of bindings supplied: This error indicates that the data types of the provided parameter bindings do not match the expected data types in the query. Ensure that the data types of the bindings match the corresponding placeholders in the query.
        # DatabaseError: file is encrypted or is not a database: This error occurs when the specified file is either encrypted or not a valid SQLite database file.
        # DatabaseError: unable to open database file: This error typically indicates that the specified database file cannot be opened, either due to incorrect file path or insufficient permissions.
        # DatabaseError: unable to open database file: This error typically indicates that the specified database file cannot be opened, either due to incorrect file path or insufficient permissions.
