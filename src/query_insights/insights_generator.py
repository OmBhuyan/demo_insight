# import datetime
import json
import logging
import os
import posixpath as pp
import traceback
from ast import literal_eval
from typing import Union

import fsspec
import pandas as pd

from .model_api import GPTModelCall
from .post_processing import append_user_query_track3, extract_code, run_insights_code
from .utils import (
    capture_stdout_to_var,
    convert_df_to_csv_string,
    get_gpt_token_count,
    rate_limit_error_handler,
    read_text_file,
    save_results_as_json,
)

MYLOGGERNAME = "QueryInsights"


class GenerateInsights:
    """Generate Insights from Tabular data extracted from Track 1(or Text to Query). It consists of three steps.

    1. Business user query to generating additional related questions that gives us insights (which we are calling it as insight questions).
    2. Generating code to answer Insight questions and original user query.
    3. Using the code result to generate summary.

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    data_config : dict
        input data_config dictionary contains the paths to the data.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    question : str
        Business user query
    dictionary : Union[list, dict]
        Data dictionary of the Track 1 data output.
    table : pd.DataFrame
        Track 1 data as a dataframe.
    skip_model : bool
        condition whether to skip the api call.
    additional_context : str, optional
        sentence which provides additional context to the original question, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        question: str,
        dictionary: Union[list, dict],
        business_overview: str,
        table: pd.DataFrame,
        output_path: str,
        sql_results_path: str,
        additional_context: str = None,
        skip_model: bool = False,
        fs=None,
    ) -> None:
        """Class constructor"""
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.question = question
        self.dictionary = dictionary
        self.business_overview = business_overview
        self.table = table
        self.additional_context = additional_context
        self.output_path = output_path

        self.connection_param_dict = user_config.connection_params
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.skip_model = skip_model

        self._fs = fs or fsspec.filesystem("file")

        # Required for decorator
        time_delay = self.user_config.time_delay
        max_retries_3a = self.model_config.table_to_insight_questions.model_params.max_tries
        max_retries_3b = self.model_config.insight_questions_to_code.model_params.max_tries
        max_retries_3c = self.model_config.summarize_insights.model_params.max_tries
        max_retries_3c2 = self.model_config.summarize_tables.model_params.max_tries

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._table_to_insight_questions = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3a
        )(self._table_to_insight_questions)
        self._insight_questions_to_code = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3b
        )(self._insight_questions_to_code)
        self._get_summary = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3c
        )(self._get_summary)
        self._get_new_summary = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3c2
        )(self._get_new_summary)

        # Init for tmp files and names
        self.tmp_file_path = pp.join(self.output_path, "tmp")
        self._fs.makedirs(self.tmp_file_path, exist_ok=True)
        self.sql_results_path = pp.join(sql_results_path, "output_table.csv")

        return

    def __repr__(self):
        try:
            full_conversation = ""
            # Track 3a
            full_conversation += "Track 3a:-\n\n"
            full_conversation += "prompt:-\n"
            full_conversation += str(self.query_to_qns.prompt)
            full_conversation += "\n"
            full_conversation += "response:-\n"
            full_conversation += (
                str(self.question_suggestion)
                + "\n"
                + str(self.question_finish)
                + "\n"
                + str(self.question_tokens)
                + "\n"
            )
            full_conversation += "-" * 100
            full_conversation += "\n\n"

            # Track 3b
            full_conversation += "Track 3b:-\n\n"
            full_conversation += "prompt:-\n"
            full_conversation += str(self.qns_to_code.prompt)
            full_conversation += "\n"
            full_conversation += "response:-\n"
            full_conversation += (
                str(self.code_suggestion)
                + "\n"
                + str(self.code_finish)
                + "\n"
                + str(self.code_tokens)
                + "\n"
            )
            full_conversation += "code result:-\n"
            full_conversation += (
                str(self.code_result) + "\nCode Error Output" + str(self.code_err_output) + "\n"
            )
            full_conversation += "-" * 100
            full_conversation += "\n\n"

            # Track 3c
            full_conversation += "Track 3c:-\n\n"
            if self.summary is not None:
                full_conversation += "prompt:-\n"
                full_conversation += str(self.summary.prompt)
                full_conversation += "\n"
                full_conversation += "response:-\n"

            full_conversation += (
                str(self.summary_suggestion)
                + "\n"
                + str(self.summary_finish)
                + "\n"
                + str(self.summary_tokens)
                + "\n"
            )
            return full_conversation
        except Exception as e:
            self.logger.info("Error - ", e)
            self.logger.info(
                "It is likely that get_insights method is not called. Call it first and rerun the print statement."
            )

    def _table_to_insight_questions(self, save_folder: str = None) -> None:
        """Business user query to generating additional related questions that gives us insights (which we are calling it as insight questions). It is otherwise known as Track 3a.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        # Load Configuration for Track 3a.
        prompt_dict = self.model_config.table_to_insight_questions.prompts
        model_param_dict = self.model_config.table_to_insight_questions.model_params

        # Initialize for the API call to GPT
        self.query_to_qns = GPTModelCall(
            prompt_dict=prompt_dict,
            question=self.question,
            additional_context=self.additional_context,
            connection_param_dict=self.connection_param_dict,
            dictionary=self.dictionary,
            business_overview=self.business_overview,
        )
        # saving the prompt
        self.logger.debug("Saving prompt for Track 3a")
        if save_folder is not None:
            with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                f.writelines("Track 3a\n")
                f.writelines(self.query_to_qns.prompt)

        # Make the API call to GPT
        (
            self.question_suggestion,
            self.question_finish,
            self.question_tokens,
        ) = self.query_to_qns.model_response(model_param_dict)

        self.logger.info(
            f"Track 3a:-\n finish token - {self.question_finish},\n token information - {self.question_tokens}"
        )

        # Saving each individual responses of Track 3 subsections as txt and json
        self.logger.debug("Saving response for Track 3a")
        if save_folder is not None:
            with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "w") as f:
                f.writelines("Track 3a\n")
                f.writelines(self.question_suggestion)
                f.write("\n" + "-" * 100 + "\n")

        return

    def _insight_questions_to_code(self, save_folder: str = None) -> None:
        """Generating code to answer Insight questions and original user query. It is otherwise known as Track 3a.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        # Append business user query to GPT generated questions.
        all_questions = append_user_query_track3(
            other_questions=self.question_suggestion, user_query=self.question
        )
        # Load Configuration for Track 3b
        prompt_dict = self.model_config.insight_questions_to_code.prompts
        model_param_dict = self.model_config.insight_questions_to_code.model_params

        # Initialize for the API call to GPT
        self.qns_to_code = GPTModelCall(
            prompt_dict=prompt_dict,
            question=all_questions,
            additional_context=self.additional_context,
            connection_param_dict=self.connection_param_dict,
            dictionary=self.dictionary,
            business_overview=self.business_overview,
        )

        if save_folder is not None:
            self.logger.debug("Saving prompt for Track 3b")
            # saving the prompt
            if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                with self._fs.open(pp.join(save_folder, "prompt.txt"), "r") as f:
                    _prompt = f.read()
            else:
                _prompt = ""

            with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                f.write(_prompt)
                f.writelines("---------------------------")
                f.writelines("\n\nTrack 3b\n")
                f.writelines(self.qns_to_code.prompt)

        # Make the API call to GPT
        self.code_suggestion, self.code_finish, self.code_tokens = self.qns_to_code.model_response(
            model_param_dict
        )
        self.logger.info(
            f"Track 3b:-\n finish token - {self.code_finish},\n token information - {self.code_tokens}"
        )

        # Post-processing
        # Trim the code to be executed.
        self.logger.info("Post processing of Track 3b started.")
        self.track3_trimmed_code = extract_code(
            string_input=self.code_suggestion, start=["<start>"], end="<end>", extract="first"
        )

        # Execute the code blocks iteratively and capture the results in stdout as str var - code_result
        # stderr as str var - code_err_output
        # fs_key is used only when fs_connection_dict.platform is not None
        self.code_result, self.code_err_output = capture_stdout_to_var(
            func=run_insights_code,
            kwargs={
                "full_code_str": self.track3_trimmed_code,
                "input_file_path": self.sql_results_path,
                "tmp_dir_path": self.tmp_file_path,
                "track3_path": self.output_path,
                "fs": self._fs,
                "fs_connection_dict": self.data_config.cloud_storage,
                "fs_key": os.getenv("BLOB_ACCOUNT_KEY"),
            },
        )
        self.logger.info("Post processing of Track 3b completed.")
        # Saving each individual responses of Track 3 subsections as txt and json
        if save_folder is not None:
            self.logger.debug("Saving code result for Track 3b")

            if self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
                with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "r") as f:
                    _track3b_responses_txt = f.read()
            else:
                _track3b_responses_txt = ""
            with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "w") as f:
                f.write(_track3b_responses_txt)
                f.writelines("Track 3b code:\n")
                f.writelines(self.code_suggestion)
                f.writelines("Track 3b code result:\n")
                f.writelines(self.code_result)
                f.write("\n" + "-" * 100 + "\n")

            with self._fs.open(pp.join(save_folder, "track3_error_responses.txt"), "w") as f:
                f.writelines("Track 3b code error:\n")
                f.writelines(self.code_err_output)

        return

    def _get_summary(self, save_folder: str = None) -> None:
        """Using the code result to generate summary. If code yielded no result due to syntax error or a blank result, a descriptive summary of the table data will be used as summary.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        if (self.skip_model) and self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
            self.logger.info("Retreiving track 3b code result from Knowledge Base.")
            with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "r") as f:
                kb_code_result = f.read()
            kb_code_result = extract_code(string_input=kb_code_result, start=["Track 3b code result:"], end="\n" + "-" * 100 + "\n", extract="first")
            if kb_code_result is not None:
                self.code_result = kb_code_result
            else:
                self.code_result = ""
        if self.code_result.strip() != "":
            self.logger.info(
                "Track 3b result was not blank, hence it's output will be used to summarize."
            )
            # Run the summary insights on code result
            prompt_dict = self.model_config.summarize_insights.prompts
            model_param_dict = self.model_config.summarize_insights.model_params

            self.summary = GPTModelCall(
                prompt_dict=prompt_dict,
                question=self.question,
                additional_context=self.additional_context,
                connection_param_dict=self.connection_param_dict,
                suggestion=self.code_result,
                dictionary=self.dictionary,
                business_overview=self.business_overview,
            )
            if save_folder is not None:
                self.logger.debug("Saving prompt for Track 3c")
                # saving the prompt
                if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                    with self._fs.open(pp.join(save_folder, "prompt.txt"), "r") as f:
                        _prompt_track3c = f.read()
                else:
                    _prompt_track3c = ""
                with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                    f.write(_prompt_track3c)
                    f.writelines("Track 3c\n")
                    f.writelines(self.summary.prompt)
            (
                self.summary_suggestion,
                self.summary_finish,
                self.summary_tokens,
            ) = self.summary.model_response(model_param_dict)
            self.logger.info(
                f"Track 3c:-\n finish token - {self.summary_finish},\n token information - {self.summary_tokens}"
            )
            # Saving each individual responses of Track 3 subsections as txt and json
            if save_folder is not None:
                self.logger.debug("Saving response for Track 3c")
                if self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
                    with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "r") as f:
                        _track3c_responses_txt = f.read()
                else:
                    _track3c_responses_txt = ""
                with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "w") as f:
                    f.write(_track3c_responses_txt)
                    f.writelines("Track 3c\n")
                    f.writelines(self.summary_suggestion)
        else:
            self._get_new_summary(token_limit=30000, save_folder=save_folder)

        return

    def _get_new_summary(self, token_limit: int = 30000, save_folder: str = None) -> None:
        """Uses the entire table (it can be either track 1 table or full data) to generate insights. If entire table is too large, it will get trimmed to fit token limit of GPT model.

        Parameters
        ----------
        token_limit : int, optional
            Token limit as defined by GPT model we are using., by default 30000
        save_folder : str, optional
            Path where all intermediate input and outputs will be saved., by default None

        Raises
        ------
        ValueError
            if any of the arguments is missing or invalid or if process errors out.
        """
        # First check whether new token limit doesnt exceed 30k (2k reserved for prompt)
        prompt_dict = self.model_config.summarize_tables.prompts
        model_param_dict = self.model_config.summarize_tables.model_params
        # Convert df to string
        table_string = convert_df_to_csv_string(self.table)

        num_tokens = int(get_gpt_token_count(input_data=table_string, model="gpt-4"))
        self.logger.info(f"Number of tokens for full table is {num_tokens}")

        curr_limit = int(self.user_config.table_top_rows)
        trimFlag = False
        self.logger.info(f"Num token = {num_tokens}, token_limit = {token_limit}")
        while num_tokens > token_limit:
            # TODO: fsspec: Not sure if this will work in Azure blob
            table_data = os.linesep.join(table_string.splitlines(keepends=True)[:curr_limit])
            num_tokens = int(get_gpt_token_count(input_data=table_data, model="gpt-4"))
            curr_limit = int(curr_limit / 2)
            trimFlag = True

        if trimFlag:
            warning_msg = f"Table has been trimmed to take top {curr_limit} rows to fit the token limit of GPT. Thus, any insights that is arises out of this data may not be correct as it's not the full representation of the data."
            self.logger.warning(warning_msg)
        else:
            warning_msg = ""

        self.logger.info(f"Number of tokens after handling for token limitation is {num_tokens}")

        if trimFlag:  # We should update this var only after exiting the loop
            table_string = table_data

        try:
            if num_tokens <= token_limit:
                # Initialize for the API call to GPT
                self.summary = GPTModelCall(
                    prompt_dict=prompt_dict,
                    question=None,
                    additional_context=None,
                    connection_param_dict=self.connection_param_dict,
                    dictionary=self.dictionary,
                    business_overview=self.business_overview,
                    table=table_string,
                )

                if save_folder is not None:
                    self.logger.debug("Saving prompt for Track 3c - table summary")
                    # saving the prompt
                    if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                        with self._fs.open(pp.join(save_folder, "prompt.txt"), "r") as f:
                            _table_prompt = f.read()
                    else:
                        _table_prompt = ""

                    with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                        f.write(_table_prompt)
                        f.writelines("Track 3c - Table summary:\n")
                        f.writelines(self.summary.prompt)

                # Make the API call to GPT
                (
                    self.summary_suggestion,
                    self.summary_finish,
                    self.summary_tokens,
                ) = self.summary.model_response(model_param_dict)

                # Prepend warning msg
                self.summary_suggestion = warning_msg + "\n\n" + self.summary_suggestion

                # Saving each individual responses of Track 3 subsections as txt and json
                if save_folder is not None:
                    self.logger.debug("Saving response for Track 3c - table summary")
                    if self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
                        with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "r") as f:
                            _track3c_responses_txt = f.read()
                    else:
                        _track3c_responses_txt = ""
                    with self._fs.open(pp.join(save_folder, "track3_responses.txt"), "w") as f:
                        f.write(_track3c_responses_txt)
                        f.writelines("Track 3c - Table summary:\n")
                        f.writelines(self.summary_suggestion)
            else:
                # Get top 100 rows
                raise ValueError("Token limit exceeded")

        except Exception as e:
            error_msg = f"Error occurred while forming summary from table. Error description:\n{e}"
            self.logger.error(error_msg)
            # self.summary = None  # For error handling
            # (self.summary_suggestion, self.summary_finish, self.summary_tokens) = (
            #     "Insights are not available.",
            #     "stop",
            #     "NA",
            # )
            raise ValueError(error_msg)

    def get_insights(self, units_to_skip=[]) -> str:
        """This is the main method that will call each subsection of track 3 in order

        Parameters
        ----------
        units_to_skip : list, optional
            Units to skip in the output when track 1 returns a scalar table. In some cases, we're seeing that the GPT returns units as `integer`, `count`, `text` etc which we want to skip. Defaulted this to empty list because we don't want this parameter to be a bottleneck for running track 2 code, by default []

        Returns
        -------
        str
            Final track 3 summary
        """
        # If we get scalar as input, no need to generate, just reply with Question and Answer as the scalar
        if self.table.shape == (1, 1):  # scalar df
            self.logger.info("Track 1 result is scalar, thus we are returning the result as is.")
            try:
                if not isinstance(self.dictionary, str):
                    self.dictionary = str(self.dictionary)
                data_dict = literal_eval(self.dictionary)
                unit = data_dict[0]["unit_of_measurement"]
                if unit.lower() in units_to_skip:
                    unit = ""
            except Exception as e:
                self.logger.error(
                    f"unit of measurement not found in data dictionary. Error description: {e}"
                )
                unit = ""

            self.summary_suggestion = (
                "Q:" + self.question + "\nAns:" + str(self.table.iloc[0, 0]) + f" {unit}"
            )
        elif self.table.shape == (1,):  # scalar series
            self.logger.info("Track 1 result is scalar, thus we are returning the result as is.")
            self.summary_suggestion = "Q:" + self.question + "\nAns:" + str(self.table.iloc[0])
        # If we have single row but many columns, pass to summary as string.
        elif self.table.shape[0] == 1 and self.table.shape[1] > 1:
            self.logger.info("Track 3 - table summary started for single row.")
            # TODO: Explore pretty prints to avoid this data sending to GPT
            self._get_new_summary(save_folder=self.output_path)
            self.logger.info("Track 3 - table summary completed for single row.")

        else:
            # CODE APPROACH
            if self.skip_model:
                # It needs to be updated.
                try:
                    self.logger.info(
                        "Question already exists. Retreiving the suggestion from Knowledge Base."
                    )
                    # run Track 3c
                    self.logger.info("Track 3c started.")
                    self._get_summary(self.output_path)
                    self.logger.info("Track 3c completed.")
                except Exception as e:
                    self.logger.error(
                        f"Error in retreiving the existing results. API call will be triggered. Error description: {e}"
                    )
                    self.logger.error(
                        traceback.format_exc()
                    )

                    self.logger.info("Track 3a started.")
                    self._table_to_insight_questions(self.output_path)
                    self.logger.info("Track 3a completed.")
                    # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                    # run Track 3b
                    self.logger.info("Track 3b started.")
                    self._insight_questions_to_code(self.output_path)
                    self.logger.info("Track 3b completed.")
                    # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                    # run Track 3c
                    self.logger.info("Track 3c started.")
                    self._get_summary(self.output_path)
                    self.logger.info("Track 3c completed.")

            else:
                self.logger.info("Track 3a started.")
                self._table_to_insight_questions(self.output_path)
                self.logger.info("Track 3a completed.")
                # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                # run Track 3b
                self.logger.info("Track 3b started.")
                self._insight_questions_to_code(self.output_path)
                self.logger.info("Track 3b completed.")
                # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                # run Track 3c
                self.logger.info("Track 3c started.")
                self._get_summary(self.output_path)
                self.logger.info("Track 3c completed.")

        # current_timestamp = datetime.datetime.now().strftime(r"%Y%m%d%H%M%S")
        self.logger.debug("Saving JSON for evaluation of Track 3.")
        file_save_path = pp.join(self.output_path, "track3_final_result.json")
        save_results_as_json(
            question=self.question,
            # additional_context=self.additional_context,
            # actual_answer="",
            predicted_answer=self.summary_suggestion,
            file_save_path=file_save_path,
            fs=self._fs,
        )

        return self.summary_suggestion
