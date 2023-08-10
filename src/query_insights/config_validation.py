# import sys
import logging
import posixpath as pp

from .pre_processing import DBConnection
from .utils import fs_connection, get_fs_and_abs_path, get_table_names, load_config

MYLOGGERNAME = "QueryInsights"


class DataConfigValidator:
    """Class that validates the data configuration

    Parameters
    ----------
    config : dict
        It should contain the paths to the input data, database, output folder, and data dictionaries.

    Raises:
        ValueError: If any of the validations fail with an invalid value.
        FileNotFoundError: If any of the file paths specified in the configuration are not found.
    """

    def __init__(self, config, fs_key: str = None):
        self.config = config
        self.fs_key = fs_key
        self.logger = logging.getLogger(MYLOGGERNAME)

    def load_fs_connection(self):
        prefix_url, storage_options = fs_connection(
            fs_connection_dict=self.config.cloud_storage, fs_key=self.fs_key
        )
        self._fs, self.paths = get_fs_and_abs_path(
            path=prefix_url, storage_options=storage_options
        )

    def validate_path(self):
        """Validates the path configuration.

        Returns:
            bool: True if the path configuration is valid.
        """
        self.input_data_path = self.config["path"]["input_data_path"]
        self.data_dictionary_path = self.config["path"]["data_dictionary_path"]
        self.output_path = self.config["path"]["output_path"]
        self.business_overview_path = self.config["path"]["business_overview_path"]
        self.api_key_location = self.config["path"]["api_key_location"]
        self.exp_name = self.config["path"]["exp_name"]
        self.database_path = self.config["db_params"]["sqlite_database_path"]

        if self.input_data_path is None or not isinstance(self.input_data_path, str):
            raise ValueError(
                f"Invalid path input_data_path: {self.input_data_path}. It should be a non-empty string."
            )

        if self.data_dictionary_path is None or not isinstance(self.data_dictionary_path, str):
            raise ValueError(
                f"Invalid path data_dictionary_path: {self.data_dictionary_path}. It should be a non-empty string."
            )

        if self.output_path is None or not isinstance(self.output_path, str):
            raise ValueError(
                f"Invalid path output_path: {self.output_path}. It should be a non-empty string."
            )

        if self.api_key_location is None or not isinstance(self.api_key_location, str):
            raise ValueError(
                f"Invalid path api_key_location: {self.api_key_location}. It should be a non-empty string."
            )

        if self.database_path is None or not isinstance(self.database_path, str):
            raise ValueError(
                f"Invalid path database_path: {self.database_path}. It should be a non-empty string."
            )

        if not isinstance(self.exp_name, str):
            raise ValueError(
                f"Invalid path exp_name: {self.exp_name}. It should be a non-empty string."
            )

        if not (
            isinstance(self.business_overview_path, str) or self.business_overview_path is None
        ):
            raise ValueError(
                f"Invalid path: {self.business_overview_path}. It should be a string or None."
            )

        try:
            if not self._fs.exists(self.input_data_path):
                raise FileNotFoundError(f"Input_data_path doesn't exist: {self.input_data_path}")

            if not self._fs.exists(self.data_dictionary_path):
                raise FileNotFoundError(
                    f"data_dictionary_path doesn't exist: {self.data_dictionary_path}"
                )
            if not self._fs.exists(self.output_path):
                raise FileNotFoundError(f"output_path doesn't exist: {self.output_path}")
            if not self._fs.exists(self.database_path):
                raise FileNotFoundError(f"database_path doesn't exist: {self.database_path}")
        except Exception as e:
            # TODO: Move this exception to individual path exists errors for better error descriptions
            self.logger.error(
                "One of the files/ folders (input, data_dictionary, output, and the database paths) not available in the Filestorage. Possibly because of broken links. Please check the slashes in the links correctly"
            )
            self.logger.error(
                f"Paths checked: {self.input_data_path}, {self.data_dictionary_path}, {self.output_path}, and {self.database_path}"
            )
            self.logger.error(str(e))
            self.error = str(e)
            return self.error

        return True

    def validate_input_file_names(self):
        """Validates the input file names.

        Returns:
            bool: True if the input file names are valid.
        """
        input_file_names = self.config["path"]["input_file_names"]

        input_data_path = self.config["path"]["input_data_path"]

        try:
            for table in list(input_file_names.keys()):
                file_name = input_file_names[table]
                file_path = pp.join(input_data_path, file_name)
                if not self._fs.exists(file_path):
                    raise FileNotFoundError(
                        f"Table-{table} not found in the expected location {file_path}"
                    )
        except Exception as e:
            # TODO: Move this exception to tables exists errors for better error descriptions
            # TODO: To remove/ change this since this config functionality is changed
            self.logger.error(
                "One or more table/s from the input_file_names are not availabile in the Filestorage. Possibly because of broken links. Please check the slashes in the links correctly"
            )
            self.logger.error(f"File names: {input_file_names.keys()}")
            self.logger.error(str(e))
            self.error = str(e)
            return self.error

        return True

    def validate_input_file_configs(self) -> bool:
        """Validates if json config is present in data dictionary for each table in database.
        Will exclude tables defined in `exclude_table_names` param

        Returns:
            bool: True if json config is present for all tables
        """

        database_connection = DBConnection(
            data_config=self.config,
            fs=self._fs,
        )
        conn = database_connection.connection_db()

        table_names = get_table_names(conn)

        data_dictionary_path = self.config["path"]["data_dictionary_path"]
        tables_to_exclude = self.config["path"]["exclude_table_names"]

        tables_to_check = list(set(table_names) - set(tables_to_exclude))

        for table in tables_to_check:
            file_config_path = pp.join(data_dictionary_path, f"{table}.json")
            if not self._fs.exists(file_config_path):
                self.logger.warning(
                    f"Data dictionary for table {table} is not found. Empty data dictionary will be used for this table."
                )
        return True

    def validate_db_params(self):
        """Validates the database parameters.

        Returns:
            bool: True if the database parameters are valid.
        """
        db_name = self.config["db_params"]["db_name"]
        if db_name is None or db_name not in ["sqlite", "mysql"]:
            raise ValueError(f"Invalid db_name: {db_name}")

        # Check if the value is an integer
        chunk_size = self.config["db_params"]["chunk_size"]
        if chunk_size is None or not isinstance(chunk_size, int):
            raise ValueError("chunk_size must be an non empty integer")

        # Check if the value is positive
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        if db_name == "mysql":
            password_path = self.config["db_params"]["password_path"]
            if password_path is None or not self._fs.exists(password_path):
                raise FileNotFoundError(f"Password file not found: {password_path}")

        return True

    def validate_cloud_storage_params(self) -> bool:
        """Validate the cloud storage parameters

        Returns
        -------
        bool
            True if cloud storage params provided are valid

        Raises
        ------
        ValueError
            "prefix_url" is empty for cloud_storage parameters
        ValueError
            "DefaultEndpointsProtocol" is empty for cloud_storage parameters
        ValueError
            "account_key_path" is empty for cloud_storage parameters when fs_key is not provided
        FileNotFoundError
            File path provided for "account_key_path" in cloud_storage parameters doesnot exist
        ValueError
            "AccountName" is empty for cloud_storage parameters
        ValueError
            "EndpointSuffix" is empty for cloud_storage parameters
        """
        cloud_storage = self.config.cloud_storage
        if cloud_storage.platform is not None:
            if cloud_storage.prefix_url is None or not bool(str(cloud_storage.prefix_url).strip()):
                raise ValueError('"prefix_url" must not be empty for cloud_storage parameters')
            if cloud_storage.DefaultEndpointsProtocol is None or not bool(
                str(cloud_storage.DefaultEndpointsProtocol).strip()
            ):
                raise ValueError(
                    '"DefaultEndpointsProtocol" must not be empty for cloud_storage parameters'
                )
            if self.fs_key is None:
                if cloud_storage.account_key_path is None:
                    raise ValueError(
                        '"account_key_path" must not be empty for cloud_storage parameters when fs_key is not provided'
                    )
                if not pp.isfile(cloud_storage.account_key_path):
                    raise FileNotFoundError(
                        f'File path provided for "account_key_path" in cloud_storage parameters doesnot exist at {cloud_storage.account_key_path}'
                    )
            if cloud_storage.AccountName is None or not bool(
                str(cloud_storage.AccountName).strip()
            ):
                raise ValueError('"AccountName" must not be empty for cloud_storage parameters')
            if cloud_storage.EndpointSuffix is None or not bool(
                str(cloud_storage.EndpointSuffix).strip()
            ):
                raise ValueError('"EndpointSuffix" must not be empty for cloud_storage parameters')

        return True

    def validate_config(self):
        """Validates the overall data configuration.

        Returns:
            Union[bool, str]: True if the configuration is valid, otherwise the error message.
        """
        try:
            self.validate_cloud_storage_params()
            self.load_fs_connection()
            self.validate_path()
            # self.validate_input_file_names()
            self.validate_input_file_configs()
            self.validate_db_params()
            return True
        except (ValueError, FileNotFoundError) as e:
            self.logger.error(str(e))
            self.error = str(e)
            # sys.exit(1)
            return self.error


class UserConfigValidator:
    """class that validates the user configuration.

    Parameters
    ----------
    config : dict
        It should contain the parameters used to interact with the OpenAI GPT-4 API to generate insights from data.
        It also includes the user interface, API call, why question threshold, table rows limit when token limit exceeds.

    Raises:
        ValueError: If any of the validations fail with an invalid value.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(MYLOGGERNAME)

    def validate_ui(self):
        """Validates the 'ui' parameter."""
        ui_enabled = self.config["ui"]
        if ui_enabled is None or not isinstance(ui_enabled, bool):
            raise ValueError("Invalid value for 'ui' parameter. It should be a boolean value.")

    def validate_skip_list(self):
        """Validates the 'skip_list' parameter."""
        skip_list = self.config["skip_list"]
        if skip_list is None or not isinstance(skip_list, list):
            raise ValueError("Invalid value for 'skip_list' parameter. It should be a list.")

        for item in skip_list:
            if not isinstance(item, str):
                raise ValueError(f"Invalid item in the list: {item}. Expected a string.")

    def validate_connection_params(self):
        """Validates the connection parameters."""
        api_type = self.config["connection_params"]["api_type"]
        if api_type is None or api_type not in ["azure", "openai"]:
            raise ValueError(f"Invalid api_type: {api_type}")

        api_base = self.config["connection_params"]["api_base"]
        if api_base is None or not isinstance(api_base, str):
            raise ValueError("Invalid value for 'api_base' parameter. It should be a string.")

        api_version = self.config["connection_params"]["api_version"]
        if api_version is None or not isinstance(api_version, str):
            raise ValueError("Invalid value for 'api_version' parameter. It should be a string.")

    def validate_user_inputs(self):
        """Validates the user input parameters."""
        question = self.config["user_inputs"]["question"]
        if not (isinstance(question, str) or question is None):
            raise ValueError(
                "Invalid value for 'question' parameter. It should be a string or None."
            )

        additional_context = self.config["user_inputs"]["additional_context"]
        if not (isinstance(additional_context, str) or additional_context is None):
            raise ValueError(
                "Invalid value for 'additional_context' parameter. It should be a string or None."
            )

        table_top_rows = self.config["table_top_rows"]
        if table_top_rows is None or not isinstance(table_top_rows, int):
            raise ValueError("table_top_rows must be an non empty integer")

        why_question_threshold = self.config["why_question_threshold"]
        # if why_question_threshold is None or not isinstance(why_question_threshold, (int, float)):
        # raise ValueError("why_question_threshold must be an integer or float")

        if why_question_threshold is not None and not (0 <= why_question_threshold <= 1):
            raise ValueError(
                "Invalid value for why_question_threshold parameter. It should be between 0 and 1."
            )

        time_delay = self.config["time_delay"]
        if time_delay is None or not isinstance(time_delay, int):
            raise ValueError("time_delay must be an non empty integer")

    def validate_config(self):
        """Validates the overall user configuration.

        Returns:
            Union[bool, str]: True if the configuration is valid, otherwise the error message.
        """
        try:
            self.validate_skip_list()
            self.validate_ui()
            self.validate_connection_params()
            self.validate_user_inputs()
            return True
        except ValueError as e:
            self.error = str(e)
            self.logger.error(str(e))
            # sys.exit(1)
            return self.error


class ModelConfigValidator:
    """A class that validates the model configuration.

    Parameters
    ----------
    config : dict
        It should contain the model_params(like engine, temperature, max_tokens...), system_role, static prompt and guidelines to follow for all the tracks

    Raises:
        ValueError: If any of the validations fail with an invalid value.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.tracks = list(self.config.keys())[:-1]

    def validate_model_params(self):
        """Validates the model parameters."""
        for i in self.tracks:
            model_params = self.config[i]["model_params"]
            engine = model_params["engine"]
            if engine is None or engine not in [
                "gpt_35_turbo",
                "gpt_test",
                "text_davinci_003",
                "text_davinci_002",
            ]:
                raise ValueError(f"Invalid engine: {engine}")

            temperature = model_params["temperature"]
            if temperature is None or not isinstance(temperature, (int, float)):
                raise ValueError(
                    "Invalid value for 'temperature' parameter. It should be a non empty number."
                )

            if not (0 <= temperature <= 1):
                raise ValueError(
                    "Invalid value for temperature parameter. It should be between 0 and 1."
                )

            max_tokens = model_params["max_tokens"]
            if not (isinstance(max_tokens, int) or max_tokens is None):
                raise ValueError(
                    "Invalid value for 'max_tokens' parameter. It should be an integer or None."
                )

            n = model_params["n"]
            if not isinstance(n, int):
                raise ValueError("Invalid value for 'n' parameter. It should be an integer.")

            stop = model_params["stop"]
            if not (isinstance(stop, str) or stop is None):
                raise ValueError(
                    "Invalid value for 'stop' parameter. It should be a string or None."
                )

            function = model_params["function"]
            if function not in ["ChatCompletion", "Completion"]:
                raise ValueError("Invalid function given")

            timeout = model_params["timeout"]
            if timeout is None or not isinstance(timeout, int):
                raise ValueError("Invalid value for 'timeout' parameter. It should be an integer.")

            max_tries = model_params["max_tries"]
            if max_tries is None or not isinstance(max_tries, int):
                raise ValueError(
                    "Invalid value for 'max_tries' parameter. It should be an integer."
                )

    def validate_prompts(self):
        """Validates the prompts."""
        for i in self.tracks:
            prompts = self.config[i]["prompts"]

            system_role = prompts["system_role"]
            if system_role is None or not isinstance(system_role, str):
                raise ValueError(
                    "Invalid value for 'system_role' parameter. It should be a non empty string."
                )

            static_prompt = prompts["static_prompt"]
            if static_prompt is None or not isinstance(static_prompt, str):
                raise ValueError(
                    "Invalid value for 'static_prompt' parameter. It should be a non empty string."
                )

            if "additional_context" in prompts:
                additional_context = prompts["additional_context"]
                if not (isinstance(additional_context, str) or additional_context is None):
                    raise ValueError(
                        "Invalid value for 'additional_context' parameter. It should be a string or None."
                    )

            if "business_overview" in prompts:
                business_overview = prompts["business_overview"]
                if not (isinstance(business_overview, str) or business_overview is None):
                    raise ValueError(
                        "Invalid value for 'business_overview' parameter. It should be a string or None."
                    )

            if "example_responses" in prompts:
                example_responses = prompts["example_responses"]
                if not (isinstance(example_responses, str) or example_responses is None):
                    raise ValueError(
                        "Invalid value for 'example_responses' parameter. It should be a string or None."
                    )

            guidelines = prompts["guidelines"]
            if not (isinstance(guidelines, str) or guidelines is None):
                raise ValueError(
                    "Invalid value for 'guidelines' parameter. It should be a string or None."
                )

        return True

    def validate_config(self):
        """Validates the overall model configuration.

        Returns:
            Union[bool, str]: True if the configuration is valid, otherwise the error message.
        """
        try:
            self.validate_model_params()
            self.validate_prompts()
            return True
        except ValueError as e:
            self.error = str(e)
            self.logger.error(str(e))
            # sys.exit(1)
            return self.error
