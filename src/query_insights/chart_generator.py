import json
import logging
import os
import posixpath as pp
import subprocess
import sys
import traceback

import fsspec
import plotly
import plotly.express as px

from .model_api import GPTModelCall

# from .utils import MyLogger
from .post_processing import (
    _get_import_statements,
    _uniquecategory_check,
    add_exception_to_code,
    clean_chart_code,
    extract_code,
)
from .utils import generate_env_dict, rate_limit_error_handler, read_text_file

MYLOGGERNAME = "QueryInsights"


class GenerateCharts:
    """
    Generate Chart type and corresponding code from data extracted from Track 1(or Text to Query). It consists of two steps.
    1. Get the chart type suggestion from GPT based on the business question and track 1's data dictionary.
    2. Using the suggestion from Track 1 and get the chart code from GPT.
    3. Using the code to generate the chart.

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    data_config : dict
        input data_config dictionary contains the paths to the data.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    question : str
        User question to be answered
    additional_context : str
        Additional context to answer the question
    table: pandas df
        Ouput from Track 1 (Text to Query)
    data_dictionary : dict
        contains table name, column name and description
    output_path : str
        path to save the results
    skip_model : bool
        condition whether to skip the api call.
    multiple_charts: bool
        condition to indicate if the user needs multiple charts
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    """

    def __init__(
        self,
        user_config,
        data_config,
        model_config,
        question,
        additional_context,
        table,
        data_dictionary,
        business_overview,
        output_path,
        skip_model: bool,
        sql_results_path: str,
        multiple_charts: bool,
        fs=None,
    ):
        """_summary_"""
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.ui = user_config.ui
        self.multiple_charts = multiple_charts

        self.question = question
        self.additional_context = additional_context
        self.data_dictionary = data_dictionary
        self.business_overview = business_overview
        self.output_path = output_path
        self.connection_param_dict = user_config.connection_params
        self.input_table = table
        self.skip_model = skip_model

        # Init for tmp files and names
        self.sql_results_path = pp.join(sql_results_path, "output_table.csv")
        self.track2_code_path = pp.join(self.output_path, "chartcode_<n>.py")
        self.track2_chart_path = pp.join(self.output_path, "chart_<n>.json")
        self.track2_metrics_path = pp.join(self.output_path, "metrics.json")

        self._fs = fs or fsspec.filesystem("file")

        # Required for decorator
        time_delay = user_config.time_delay
        max_retries_2a = model_config.query_to_chart_type.model_params.max_tries
        max_retries_2b = model_config.query_to_chart_code.model_params.max_tries

        self.logger = logging.getLogger(MYLOGGERNAME)

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._charttype_apicall = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_2a
        )(self._charttype_apicall)
        self._chartcode_apicall = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_2b
        )(self._chartcode_apicall)
        return

    def _charttype_apicall(self):
        """Track 2a - Using Business user query and Track 1 (SQL query) results to get a chart-type suggestion from the model."""
        # Load Configuration for Track 2a.
        prompt_dict = self.model_config.query_to_chart_type.prompts
        model_param_dict = self.model_config.query_to_chart_type.model_params

        # Updating the prompts based on multiple charts flag.
        if self.multiple_charts:
            prompt_dict["static_prompt"] = prompt_dict["static_prompt_multiplecharts"]

        self.query_to_charttype_ins = GPTModelCall(
            prompt_dict=prompt_dict,
            question=self.question,
            additional_context=self.additional_context,
            connection_param_dict=self.connection_param_dict,
            dictionary=self.data_dictionary,
            business_overview=self.business_overview,
        )
        # GPT Model call for Track 2a
        (
            self.chart_type_output,
            self.chart_type_finish,
            self.chart_type_tokens,
        ) = self.query_to_charttype_ins.model_response(model_param_dict)

        return

    def _chartcode_apicall(self):
        """Track 2b - Using Business user query and Track 1 (SQL query) results and Track 2a (Chart Type Suggestion)
        to get a chart-code suggestion from the model."""
        # Load Configuration for Track 2b.
        prompt_dict = self.model_config.query_to_chart_code.prompts
        model_param_dict = self.model_config.query_to_chart_code.model_params

        # Updating the prompts based on multiple charts flag.
        if self.multiple_charts:
            prompt_dict["static_prompt"] = prompt_dict["static_prompt_multiplecharts"]
            prompt_dict["guidelines"] = prompt_dict["guidelines_multiplecharts"]

        self.query_to_chartcode_ins = GPTModelCall(
            prompt_dict=prompt_dict,
            question=self.question,
            additional_context=self.chart_type_output,
            connection_param_dict=self.connection_param_dict,
            dictionary=self.data_dictionary,
            business_overview=self.business_overview,
        )

        # GPT model call for Track 2b
        (
            self.chart_code_output,
            self.chart_code_finish,
            self.chart_code_tokens,
        ) = self.query_to_chartcode_ins.model_response(model_param_dict)

        return

    def _post_processing(self, track: str):
        """
        Save the files based on the track details.

        Parameters
        ----------
        track : str
            Either Track2a or Track2b
        """
        if track == "Track2a":
            # Save the prompt
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="charttype_prompt.txt",
                content=self.query_to_charttype_ins.prompt,
            )

            # Save the chart-type suggestion
            self.logger.debug("Saving chart type suggestion.")
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="charttype_suggestion.txt",
                content=self.chart_type_output,
            )

        elif track == "Track2b":
            # Save the prompt
            self.logger.debug("Saving chart code suggestion prompt.")
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="chartcode_prompt.txt",
                content=self.query_to_chartcode_ins.prompt,
            )
            # Save the chart-code suggestion
            self.logger.debug("Saving chart code suggestion.")
            if self.question is None:
                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name="chartcode_suggestion_wo_question.txt",
                    content=self.chart_code_output,
                )
            else:
                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name="chartcode_suggestion.txt",
                    content=self.chart_code_output,
                )

            # Extracting the codes from the suggestion and executing it.
            # If there are more than one start substrings and one element is a substring of other, please order it in a way the first element is a subset of other element.
            # For example - If the start elements are <start> and <begin>, they can be given in any order.
            # If start elements are "```python" and "```", one is a substring of the other. "```" should be specified before "```python".
            self.chart_code_list = extract_code(
                string_input=self.chart_code_output,
                start=["```", "```python"],
                end="```",
                extract="all",
            )

            # Get all the import statements from the initial code.
            import_statements = _get_import_statements(self.chart_code_list[0])
            # Post process all the codes in the for loop and create separate chart code files.
            for i in range(0, len(self.chart_code_list)):
                self.chart_code = self.chart_code_list[i]
                self.chart_code = self.chart_code.replace("metrics.json", self.track2_metrics_path)

                # Update the file names based on the iteration.
                # Suffix is not necessary for the first code.
                if i == 0:
                    track2_chart_path = self.track2_chart_path.replace("_<n>", "")
                    track2_code_file = "chartcode.py"
                    add_import_statements = None
                else:
                    track2_chart_path = self.track2_chart_path.replace("<n>", str(i))
                    track2_code_file = "chartcode_" + str(i) + ".py"
                    add_import_statements = import_statements

                # Post process the code
                # fs_key is used only when fs_connection_dict.platform is not None
                self.chart_code = clean_chart_code(
                    full_code_str=self.chart_code,
                    input_file_path=self.sql_results_path,
                    chart_save_path=track2_chart_path,
                    import_statements=add_import_statements,
                )

                self.chart_code = add_exception_to_code(
                    full_code_str=self.chart_code,
                    include_pattern=("fig."),
                    exclude_pattern=("fig.write_json"),
                )
                self.logger.debug(f"Saving chart code to {self.track2_code_path}")
                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name=track2_code_file,
                    content=self.chart_code,
                )

    def _run_chart_code(self, file_suffix):
        """
        The python code is executed. The code will create either plotly fig object or Metrics JSON.
        If the code has plotly fig object, it is saved as html and png files.
        Finally the JSON is returned from the function (Plotly fig JSON or Metrics JSON).

        Returns
        -------
        fig
            Figure Object / Metrics object - JSON
        """
        # Update the file names based on the iteration.
        # Suffix is not necessary for the first code.
        if file_suffix == "0":
            track2_code_path = self.track2_code_path.replace("_<n>", "")
            track2_chart_path = self.track2_chart_path.replace("_<n>", "")
            chart_html_filename = "chart.html"
            chart_png_filename = "chart.png"
        else:
            track2_code_path = self.track2_code_path.replace("<n>", file_suffix)
            track2_chart_path = self.track2_chart_path.replace("<n>", file_suffix)
            chart_html_filename = "chart_" + file_suffix + ".html"
            chart_png_filename = "chart_" + file_suffix + ".png"

        # Run the chart code using subprocess.
        # subprocess_args = ["python", self.track2_code_path]

        # sys.executable is used as sometimes when setting env variables
        # python path is not being recognized
        subprocess_args = [sys.executable, "-c"]

        try:
            with self._fs.open(track2_code_path, "r") as f:
                code = f.read()
            subprocess_args.append(code)

            # Generate env dictionary from cloud storage parameters to be passed to subprocess as env variables
            # env will be None in case no cloud storage parameters are specified
            env = generate_env_dict(
                cloud_storage_dict=self.data_config.cloud_storage,
                account_key=os.getenv("BLOB_ACCOUNT_KEY"),
            )

            # subprocess output is of type - https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess
            subprocess_result = subprocess.run(
                args=subprocess_args,
                text=True,
                capture_output=True,  # For populating stderr
                env=env,  # Sets environment variables to run the code
            )

            self.logger.debug(f"Chart code run return status = {subprocess_result.returncode}")
            subprocess_result.check_returncode()

            # If code runs successfully, code will resume running below statements
            if "metrics.json" in self.chart_code_list[int(file_suffix)]:
                with self._fs.open(self.track2_metrics_path, "r") as file:
                    metrics_json = file.read()
                # Parse the JSON data into a dictionary
                self.return_object = {"Chart Metrics": json.loads(metrics_json)}
            else:
                with self._fs.open(track2_chart_path, mode="r") as fp:
                    self.return_object = plotly.io.read_json(fp)

                self.logger.debug("Saving chart.")
                self._save_outputs(
                    file_type="plotly_fig",
                    output_folder=self.output_path,
                    file_name=[chart_html_filename, chart_png_filename],
                    content=self.return_object,
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error while running the chart code. Error description:\n{e.stderr}"
            )
            self.return_object = None
            raise ValueError(e.stderr)
        except Exception as e:
            self.logger.error(f"Error while running the chart code. Error description:\n{e}")
            self.return_object = None
            raise ValueError(e)

        return self.return_object

    def _save_outputs(self, file_type: str, output_folder: str, file_name: str, content) -> None:
        """
        Save the outputs in the respective folders based on file types.
        """
        if file_type == "text":
            # save chart code suggestion
            with self._fs.open(pp.join(output_folder, file_name), "w") as f:
                f.writelines(content)
                # exec(self.chart_code_suggestion)
                # fig.write_image(os.path.join(output_path_upd, "chart.jpeg"))
        elif file_type == "plotly_fig":
            # save chart figure as html
            # TODO: Add a comment indicating it should be a list with 0th index needs to be plotly and 1st is a png (in the place where this is called)
            with self._fs.open(pp.join(output_folder, file_name[0]), mode="w") as fp:
                plotly.io.write_html(content, file=fp)
            with self._fs.open(pp.join(output_folder, file_name[1]), mode="wb") as fp:
                # content.write_image(fp)
                plotly.io.write_image(content, file=fp)

            # As we are already saving in code, no need to overwrite it.
            # content.write_json(os.path.join(output_folder, "chart.json"), pretty=True)

    def _get_chart_object(self) -> list:
        """
        Run chart code(s) and return chart objects.

        Returns
        -------
        list
            Chart Object - JSON
        """
        self._chartcode_apicall()
        self._post_processing(track="Track2b")
        chart_object = []
        for i in range(0, len(self.chart_code_list)):
            chart_object.append(self._run_chart_code(str(i)))

        return chart_object

    def _get_chart_suggestion(self) -> None:
        """
        Model call and post processing for Chart code suggestion - Track 2b.

        Returns
        -------
        fig
            Chart Object - JSON
        """
        try:
            self.logger.info("Track 2a started.")
            # Getting the chart type suggestion.
            self._charttype_apicall()
            # Saving the chart type results.
            self._post_processing(track="Track2a")
            self.logger.info("Track 2a completed")

            self.logger.info("Track 2b started.")
            chart_object = None
            try:
                self.logger.info("Trying an iteration with the user question.")
                chart_object = self._get_chart_object()
            except Exception as e:
                error_string = f"Iteration with question returned an error. Error - {e}"
                self.logger.info(error_string)
                self.logger.info("Trying an iteration without the user question.")
                self.question = None
                self.additional_context = None
                chart_object = self._get_chart_object()

            self.logger.info("Track 2b completed.")
        except Exception as e:
            error_msg = f"""
            Error while generating Chart Type/Code Suggestion, error: {e}
            """
            self.logger.error(error_msg)
            self.logger.error(f"Error Traceback:\n{traceback.format_exc()}")
            raise ValueError(error_msg)

        return chart_object

    def process_suggestion(self):
        """
        This is the main function which runs the Track 2 process in order.

        Returns
        -------
        fig
            Figure Object - JSON if it's a success.
            None - if Track 2 is skipped if the input table passed has just 1 row.
        """
        chart_object = None
        # Check if the output from Track 1 is suitable to generate a chart.
        # If the output table has only one row/scalar value, then it is not necessary to generate chart.
        if self.input_table.shape[0] <= 1:
            self.logger.info(
                "The output from Track 1 has just 1 row, which is not suitable to generate a chart."
            )
        # If the output table has only one column, check whether it is an ID or unique categorical column.
        elif self.input_table.shape[1] == 1:
            self.uniqueCategory_flag = _uniquecategory_check(self.input_table)
            # If it's an ID column or unique categorical column, can skip the histogram since it is not needed.
            if self.data_dictionary["columns"][0]["id"] == "Yes" or self.uniqueCategory_flag:
                self.logger.info(
                    """Output from track 1 has just one ID column or unique categorical column.
                    List or Table view should be appropriate."""
                )
            else:
                # If it's not an ID or unique categorical column, try creating a histogram.
                try:
                    df = self.input_table.copy()
                    # print(type(self.data_dictionary['columns'][0]['column_name']))
                    fig = px.histogram(df, x=self.data_dictionary["columns"][0]["column_name"])
                    fig.update_layout(
                        xaxis_title=self.data_dictionary["columns"][0]["description"].title(),
                        yaxis_title="Frequency",
                        title="Distribution of Values",
                    )
                    # fig.show()
                    self._save_outputs(
                        file_type="plotly_fig",
                        output_folder=self.output_path,
                        file_name=["chart.html", "chart.json"],
                        content=fig,
                    )
                    chart_object = [fig]
                except Exception as e:
                    error_msg = f"""Output from track 1 has just one column. List or Table view should be appropriate.
                    If the output has more rows, use the 'Export Output as csv' option.
                    Error while generating Chart Type/Code Suggestion, error: {e}
                    """
                    self.logger.error(error_msg)
                    self.logger.error(f"Error Traceback:\n{traceback.format_exc()}")
                    raise ValueError(error_msg)
        else:
            if self.skip_model:
                try:
                    self.logger.info(
                        "Question already exists in the Knowledge base. Extracting the code."
                    )
                    # Get the list of chart code files from the Track 2 path.
                    file_list = self._fs.ls(self.output_path)
                    chart_codes = sorted(
                        pp.basename(file)
                        for file in file_list
                        if pp.basename(file).startswith("chartcode")
                        and pp.basename(file).endswith(".py")
                    )

                    # Read the chart code(s) from the Track 2 path.
                    self.chart_code_list = []
                    for i in range(0, len(chart_codes)):
                        chart_code = chart_codes[i]

                        chartcode_path = pp.join(self.output_path, chart_code)
                        chartcode_str = read_text_file(chartcode_path, fs=self._fs)
                        # TODO: clean chart code function here is only relevant for old KB files. can be removed after KB is updated
                        if i == 0:
                            track2_chart_path = self.track2_chart_path.replace("_<n>", "")
                            track2_file_name = "chartcode.py"
                        else:
                            track2_chart_path = self.track2_chart_path.replace("<n>", str(i))
                            track2_file_name = f"chartcode_{i}.py"

                        if "df = pd.read_csv" not in chartcode_str:
                            self.logger.info("cleaning the chart code present in knowledge base")
                            # fs_key is used only when fs_connection_dict.platform is not None
                            chartcode_str = clean_chart_code(
                                chartcode_str,
                                self.sql_results_path,
                                chart_save_path=track2_chart_path,
                                import_statements=None,
                            )
                            self._save_outputs(
                                file_type="text",
                                output_folder=self.output_path,
                                file_name=track2_file_name,
                                content=chartcode_str,
                            )
                        self.chart_code_list.append(chartcode_str)

                    # Loop through the chart codes and get the final chart object (JSON).
                    chart_object = []
                    for i in range(0, len(self.chart_code_list)):
                        chart_object.append(self._run_chart_code(str(i)))
                    self.logger.info("Chart generated using existing code from Knowledge base.")
                except Exception as e:
                    self.logger.info(
                        f"Error in retreiving details from Knowledge base. API Call will happen. Error - {e}"
                    )
                    chart_object = self._get_chart_suggestion()
            else:
                chart_object = self._get_chart_suggestion()

        return chart_object
