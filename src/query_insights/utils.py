import functools
import io
import json
import logging
import os
import posixpath as pp
import re
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout, suppress
from functools import partial
from typing import Callable, Tuple, Union

import fsspec
import nltk
import openai
import pandas as pd
import spacy
import tiktoken
import wrapt
import wrapt_timeout_decorator
import yaml

MYLOGGERNAME = "QueryInsights"


class CloudStorageHandler(logging.Handler):
    """
    Logging FileHandler for cloud storage
    """

    def __init__(self, fs, log_file_path: str):
        """
        Parameters
        ----------
        fs : fsspec.filesystem, optional
            Filesystem of the url
        log_file_path : str
            File path to save the logs
        """
        super().__init__()
        self._fs = fs
        self._log_file_path = log_file_path

    def emit(self, record):
        log_message = self.format(record)
        try:
            # Load previous log messages
            with self._fs.open(self._log_file_path, "r") as f:
                _history = f.read()
        except FileNotFoundError:
            # If file doesn't exist yet, history is empty
            _history = ""
        # Write old log messages and new log messages
        with self._fs.open(self._log_file_path, "w") as f:
            f.write(_history)
            f.write(log_message + "\n")


def create_logger(
    logger_name: str = "QueryInsights",
    level: str = "WARNING",
    log_file_path: str = None,
    verbose: bool = True,
    fs=None,
) -> None:
    """Creates logger object. By default, logger objects have global namespace.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger. Used by other scripts in this package, by default "QueryInsights"
    level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"
    log_file_path : str, optional
        File path to save the logs, by default None
    verbose : bool, optional
        If `True` logs will be printed to console, by default True
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    """
    # Create logger
    logger = logging.getLogger(logger_name)

    # Set level
    all_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level in all_levels:
        logger.setLevel(level)
    else:
        print(
            f"""{level} is not part of supported levels i.e.('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            Setting WARNING as default level"""
        )
        logger.setLevel("WARNING")

    # Set handler
    logger.handlers.clear()
    if log_file_path is not None:
        # TODO: fsspec: Check for other cloud storage providers like S3
        fs = fs or fsspec.filesystem("file")
        if fs.protocol == "file":
            if not fs.exists(log_file_path) and not fs.exists(os.path.dirname(log_file_path)):
                fs.makedirs(os.path.dirname(log_file_path))
            fh = logging.FileHandler(log_file_path, mode="w")
        else:
            if not fs.exists(log_file_path) and not fs.exists(os.path.dirname(log_file_path)):
                # fs.makedirs is not creating a directory in blob storage sometimes
                # Create a dummy file if it doesn't exist
                with fs.open(log_file_path, mode="w") as _:
                    pass
            fh = CloudStorageHandler(fs, log_file_path)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler()  # (sys.stdout)
        sh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(sh)
    if log_file_path is None and not verbose:
        logger.addHandler(logging.NullHandler())

    return


def load_sqlite3_database(
    path: str, logger_name: str = "QueryInsights", fs=None
) -> sqlite3.Connection:
    """
    Loads sqlite3 database from the given path.
    Loads sqlite3 database directly if file system is local.
    For other file systems, a temporary file will get created in local filesystem corresponding to the sqlite3 database and it will be loaded from there.

    Parameters
    ----------
    path : str
        Path to the sqlite3 database
    logger_name : str
        Logger name, by default ``QueryInsights``
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``

    Returns
    -------
    sqlite3.Connection
        Connection object to the database.
    """
    # Create logger
    logger = logging.getLogger(logger_name)

    fs = fs or fsspec.filesystem("file")
    logger.info("Loading Sqlite3 database")
    if fs.protocol == "file":
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
        except Exception as e:
            logger.error(f"Error occured when loading sqlite3 database, {e}")
            logger.error(traceback.format_exc())
            raise ValueError("Unable to load sqlite3 db")
        logger.info("Sqlite3 database is loaded")
    else:
        tmp_path = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        if os.name == "nt":
            temp_path = tempfile.TemporaryDirectory()
            # TODO: fsspec: fs.copy not working in windows, so using fs.get (which is working in Windows and Linux) which we need to test fs.get upon deploying to azure functions
            # https://filesystem-spec.readthedocs.io/en/latest/copying.html
            fs.get(path, os.path.join(temp_path.name, os.path.basename(temp_path.name)))
            try:
                _conn = sqlite3.connect(
                    os.path.join(temp_path.name, os.path.basename(temp_path.name)),
                    check_same_thread=False,
                )
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                _conn.backup(conn)
                _conn.close()
            except Exception:
                logger.error("Error occured when loading sqlite3 database in Windows")
                logger.error(traceback.format_exc())
                shutil.rmtree(temp_path.name)
                raise ValueError("Unable to load sqlite3 db")
            logger.info("Sqlite3 database is loaded")
            shutil.rmtree(temp_path.name)
        else:
            with fs.open(path, "rb") as f:
                # TODO: Do more research on tempfiles and why the below commented code isn't working in windows. If we can make this work, we can use the same in track 3 also
                with tempfile.NamedTemporaryFile(suffix=".db") as fp:
                    fp.write(f.read())
                    try:
                        conn = sqlite3.connect(fp.name, check_same_thread=False)
                    except Exception:
                        logger.error("Error occured when loading sqlite3 database")
                        logger.error(traceback.format_exc())
                        raise ValueError("Unable to load sqlite3 db")
            logger.info("Sqlite3 database is loaded")

    return conn


class TokenLimitError(Exception):
    """custom exception class to indicate GPT token limit is exceeded."""

    pass


class SensitiveContentError(Exception):
    """custom exception class to indicate openAI model's flag the content as sensitive."""

    pass


class TimeoutError(Exception):
    """Custom exception class to indicate we got timeout error even after retry."""

    pass


class DotifyDict(dict):
    """
    DotifyDict makes a dict accessable by dot(.)
    for a dictionary ex = {'aaa': 1, 'bbb': {'ccc': 2}}, ex['bbb']['ccc'] can be accessed using ex.bbb.ccc

    Returns
    -------
    a custom dict type accessable by dot(.)
    """

    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotifyDict):
            value = DotifyDict(value)
        super(DotifyDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotifyDict.MARKER)
        if found is DotifyDict.MARKER:
            found = DotifyDict()
            super(DotifyDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


# TODO: this needs to be changed as it signal alarm is for Posix only.
def timeout_depricated(func):
    """
    Decorator to timeout a function call.

    Parameters
    ----------
    func : function
        function to be decorated

    Returns
    -------
    function
        decorated function

    Raises
    ------
    TimeoutError
        if the function call times out.
    """

    def decorator(*args, **kwargs):
        """_decorator_

        Parameters
        ----------
        *args : list
            list of arguments
        **kwargs : dict
            dictionary of keyword arguments

        Returns
        -------
        function
            decorated function

        Raises
        ------
        TimeoutError
            if the function call times out.
        """
        model_param_dict = kwargs.get("model_param_dict", {})
        seconds = model_param_dict.get("timeout")
        max_tries = model_param_dict.get("max_tries", 1)
        error_message = "Function call timed out"
        if os.name == "nt":
            # Its windows, skip this handling
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise e

            return result
        else:
            for i in range(max_tries):
                if seconds is None:
                    return func(*args, **kwargs)
                else:

                    def _handle_timeout(signum, frame):
                        raise TimeoutError(error_message)

                    def _handle_exception(exc_type, exc_value, traceback):
                        if i == max_tries - 1:
                            raise exc_type(exc_value).with_traceback(traceback)

                    signal.signal(signal.SIGALRM, _handle_timeout)
                    signal.alarm(seconds)
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        _handle_exception(type(e), e, e.__traceback__)
                    finally:
                        signal.alarm(0)
            return result
        # raise TimeoutError(error_message)

    return decorator


def timeout(timeout_seconds: int = 2, max_tries: int = 2, use_signals: bool = False):
    """
    Decorator to timeout a function call.

    Parameters
    ----------
    timeout_seconds : int
        Seconds to timeout in, by default 2
    max_tries : int
        Number retires after timeout, by default 2
    use_signals : bool
        If signals module is to be used,
        will raise error if not run in main thread,
        defaults to multiprocessing if False, by default False

    Returns
    -------
    function
        decorated function

    Raises
    ------
    TimeoutError
        if the function call times out.
    """

    @wrapt.decorator
    def decorator(wrapped, instance, args, kwargs):
        """Decorator function

        Parameters
        ----------
        wrapped :
            Function that is being wrapped in the decorator
        instance :
            Instance object
        args : list
            List of arguments
        kwargs : dict
            Dictionary of keyword arguments

        Returns
        -------
        function
            Decorated function

        Raises
        ------
        TimeoutError
            If the function call times out.
        """
        # TODO: Add logger here to log timeouts, errors and retries

        if os.name == "nt":
            # TODO: Needs more debugging to make it work in windows
            # Its windows, skip this handling
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                raise e

            return result

        model_param_dict = kwargs.get("model_param_dict", {}) or (args[0] if args else {})
        _seconds = model_param_dict.get("timeout", None)
        _max_tries = model_param_dict.get("max_tries", None)

        # Track number of retries
        retries = 0
        # Give preference to timeout provided in config file over default timeout
        timeout_seconds_ = _seconds if _seconds is not None else timeout_seconds
        # Give preference to max tries provided in config file over default max tries
        max_tries_ = _max_tries if _max_tries is not None else max_tries
        # While number of retires is lesser than value provided, the function is run max_tries_ times
        while retries < (max_tries_):
            try:
                # Timeout wrapper on the specified function
                result = wrapt_timeout_decorator.timeout(
                    timeout_seconds_,
                    use_signals=use_signals,
                    timeout_exception=TimeoutError,
                )(wrapped)(*args, **kwargs)
                return result
            except TimeoutError:
                # Except TimeoutError add to number of retries
                retries += 1

            except Exception:
                # Code errored out due to issues in function
                # TODO: Log these errors as they are coming from within the function being decorated
                retries += 1
                # print(traceback.format_exc())

        raise TimeoutError(f"Function {wrapped.__name__} timed out after {max_tries_} retries.")

    return decorator


def fs_connection(fs_connection_dict=None, fs_key=None):
    """Returns the prefix url and storage options by reading the connection dictionary and the account key

    Parameters
    ----------
    fs_connection_dict : dict, optional
        Dictionary containing configuration settings to connect to the cloud, by default None
    fs_key : str, optional
        Account key to make connection to the specified platform (in fs_connection_dict). If platform is not None, it will look for the path in the data_config and read the key from there. Can be left as None for using local File storage (Windows, Linux) (when platform in None), by default None

    Returns
    -------
    str
        Prefix URL for connecting to the file storage. None for normal FS (Linux, Windows etc)
    str
        Storage options for connecting to the file storage. None for normal FS (Linux, Windows etc)

    Raises
    ------
    ValueError
        When platform is not None and appropriate account key is not specified to make the connection
    """
    platform = fs_connection_dict.platform
    if platform is None:
        prefix_url = None
        storage_options = None
    elif platform == "azure":
        # TODO: fsspec: Add validations to make sure that this path exists and add some logger messages?
        # load Azure account key
        if fs_key is not None:
            load_key_to_env(secret_key=fs_key, env_var="BLOB_ACCOUNT_KEY")
        else:
            load_key_to_env(
                secret_key=fs_connection_dict.account_key_path, env_var="BLOB_ACCOUNT_KEY", fs=None
            )
        account_key = os.getenv("BLOB_ACCOUNT_KEY")
        if account_key is None:
            raise ValueError(
                "Failed loading the Azure account key into environment variable. Please use `fs_key` parameter pass account key as a string or specify the path to the account key in the data configuration (data_config.cloud_storage.account_key_path)"
            )

        prefix_url = fs_connection_dict.prefix_url
        fs_connection_string = f"DefaultEndpointsProtocol={fs_connection_dict.DefaultEndpointsProtocol};AccountName={fs_connection_dict.AccountName};AccountKey={account_key};EndpointSuffix={fs_connection_dict.EndpointSuffix}"
        storage_options = {
            "connection_string": fs_connection_string,
            "account_key": account_key,
        }
    return prefix_url, storage_options


def get_fs_and_abs_path(path, storage_options=None):
    """Get the Filesystem and paths from a urlpath and options.

    Parameters
    ----------
    path : string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data. If None is provided, this
        parameter will be defaulted to an empty string to avoid errors.
    storage_options : dict, optional
        Additional keywords to pass to the filesystem class.

    Returns
    -------
    fsspec.FileSystem
       Filesystem Object
    list(str)
        List of paths in the input path.
    """
    path = path if path is not None else ""
    fs, _, paths = fsspec.core.get_fs_token_paths(path, storage_options=storage_options)
    if len(paths) == 1:
        return fs, paths[0]
    else:
        return fs, paths


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionary of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)


def load_config(cfg_file, fs=None):
    """Create the Context from a config file location path.

    Parameters
    ----------
    path : str
        Location path of the .yaml config file.
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    Dotified dictionary with all the config parameters.
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(cfg_file):

        def _dotted_access_getter(key, dct):
            for k in key.split("."):
                dct = dct[k]
            return dct

        def _repl_fn(match_obj, getter):
            return getter(match_obj.groups()[0])

        def _interpolate(val, repl_fn):
            if isinstance(val, dict):
                return {k: _interpolate(v, repl_fn) for k, v in val.items()}
            elif isinstance(val, list):
                return [_interpolate(v, repl_fn) for v in val]
            elif isinstance(val, str):
                # We shouldn't replace slashes in url's/ links. In out case api_base link is a HTTP and we shouldn't replace `/` with os specific slash in it
                # if not validators.url(val):
                #     val = val.replace(pp.sep, os.path.sep)
                return re.sub(r"\$\{([\w|.]+)\}", repl_fn, val)
            else:
                return val

        cfg = load_yml(cfg_file, fs=fs)

        cfg = _interpolate(
            cfg,
            partial(_repl_fn, getter=partial(_dotted_access_getter, dct=cfg)),
        )
        cfg["config_file"] = pp.abspath(cfg_file)

        return DotifyDict(cfg)
    else:
        raise ValueError(f"{cfg_file} is not a valid config file.")


def load_key_to_env(secret_key, env_var, fs=None):
    """
    Loads the the secret key into specified environment variables

    Parameters
    ----------
    secret_key : str
        This can be a secret key as a string or a path to a file that contains the secret key
    env_var : str
        Environment variable to which the secret key will be loaded
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(secret_key):
        # If we received file path, read the file, else directly pass in the key.
        with fs.open(secret_key, "r") as f:
            key = f.read()
    else:
        key = secret_key

    os.environ[env_var] = key


def load_db_credentials(password_path, fs=None):
    """
    Loads the database password into environment variables

    Parameters
    ----------
    password_path : str
        this can be the path to a txt file that contains password, or the password itself
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(password_path):
        # If we received file path of password, read the file, else directly pass in the password.
        with fs.open(password_path, "r") as f:
            password = f.read()
    else:
        password = password_path
    return password


def read_data(
    path,
    logger_name: str = "QueryInsights",
    fs=None,
):
    """
    Reads the excel/csv file and formats the column names.

    Parameters
    ----------
    path : str
        path to the csv or xlsx file
    logger_name : str
        Logger name, by default ``QueryInsights``
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        formatted data frame that can be loaded to SQL DB

    Raises
    ------
    ValueError
        if the path is not a valid csv or xlsx file.
    """
    # Create logger
    # TODO: All file reads should happen using this function. Please add modifications for other file formats as required.
    logger = logging.getLogger(logger_name)

    try:
        fs = fs or fsspec.filesystem("file")
        if path.endswith("csv"):
            with fs.open(path, mode="rb") as fp:
                data = pd.read_csv(fp)
        if path.endswith("xlsx"):
            with fs.open(path, mode="rb") as fp:
                data = pd.read_excel(fp, engine="openpyxl")
    except Exception as e:
        logger.error(f"Error:{e}")
        raise ValueError(f"{path} is not a valid csv or xlsx file.")

    return data


def read_and_process_data(
    path,
    logger_name: str = "QueryInsights",
    fs=None,
    **kwargs,
):
    """
    Reads the excel/csv file and formats the column names and the date columns to load it to a SQL database

    Parameters
    ----------
    path : str
        path to the csv or xlsx file
    logger_name : str
        Logger name, by default ``QueryInsights``
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        formatted data frame that can be loaded to SQL DB

    Raises
    ------
    ValueError
        if the path is not a valid csv or xlsx file.
    """

    # Create logger
    logger = logging.getLogger(logger_name)

    try:
        fs = fs or fsspec.filesystem("file")
        if path.endswith("csv"):
            # TODO: Changed this back since it isn't working in Windows. Not sure if this will work with Azure blob or not
            # data = pd.read_csv(prefix_url+path, storage_options=storage_options)
            with fs.open(path, mode="rb") as fp:
                data = pd.read_csv(fp, **kwargs)
        if path.endswith("xlsx"):
            with fs.open(path, mode="rb") as fp:
                data = pd.read_excel(fp, engine="openpyxl", **kwargs)
    except Exception as e:
        logger.error(f"Error:{e}")
        raise ValueError(f"{path} is not a valid csv or xlsx file.")
    pattern = re.compile(r"[^\w]")

    data.columns = [col.lower() for col in data.columns]
    data.columns = [col.replace(" ", "_") for col in data.columns]
    data.columns = [pattern.sub("_", col) for col in data.columns]

    # TODO Need to have generalised way to convert datetime columns

    for col in data.columns:
        try:
            data[col] = pd.to_datetime(data[col], format="%d-%m-%Y %H:%M:%S").astype(str)
        except Exception as e:
            logger.debug(
                f"Error while formatting the column {col} with the format '%d-%m-%Y %H:%M:%S',\nError description:{e}"
            )
            # print(f"error {e} occured while converting {col} to datetime")
            try:
                data[col] = pd.to_datetime(data[col], format="%d-%m-%Y %H:%M").astype(str)
            except Exception as e:
                logger.debug(
                    f"Error while formatting the column {col} with the format '%d-%m-%Y %H:%M',\nError description:{e}"
                )
                # print(f"error {e} occured while converting {col} to datetime")
                try:
                    data[col] = pd.to_datetime(data[col], format="%d-%m-%Y").astype(str)
                except Exception as e:
                    logger.debug(
                        f"Error while formatting the column {col} with the format '%d-%m-%Y',\nError description:{e}"
                    )
                    # logger.error(f"error {e} occured while converting {col} to datetime")
                    pass

    return data


def read_and_process_data_dictionary(data_dict):
    """
    edit the column names in the data dictionary so that they match with the formatted data frame

    Parameters
    ----------
    data_dict : dict
        contains data dictionary with column names and their descriptions

    Returns
    -------
    dict
        data dictionary with updated column names
    """
    pattern = re.compile(r"[^\w]")
    for column in data_dict["columns"]:
        column["name"] = pattern.sub("_", column["name"].lower().replace(" ", "_"))
    return data_dict


def load_data_dictionary(path, fs=None, **kwargs):
    """
    read the data dictionary JSON files

    Parameters
    ----------
    path : str
        path to the JSON file
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        data dictionary that can be used in prompts

    Raises
    ------
    ValueError
        if the path is not a valid JSON file.
    """
    try:
        fs = fs or fsspec.filesystem("file")
        with fs.open(path, "r") as f:
            data_dictionary = json.load(f, **kwargs)
    except Exception as e:
        print(f"error {e} occured while reading the data dictionary")
    return read_and_process_data_dictionary(data_dictionary)


def load_to_in_memory_db(df, table_name, conn):
    """
    Loads the data frame to an in memory SQLite database

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be loaded
    table_name : str
        name of the table to be created
    conn : sqlite3.Connection
        connection to the database

    Returns
    -------
    None
    """
    df.fillna("None", inplace=True)

    column_names = df.columns.tolist()

    sql_data_types = {
        "object": "TEXT",
        "int64": "INTEGER",
        "float64": "REAL",
        "datetime64[ns]": "TEXT",
        "bool": "INTEGER",
    }

    columns = ",\n".join(
        [
            f"{col_name} {sql_data_types[str(df[col_name].dtype)]} DEFAULT NULL"
            for col_name in column_names
        ]
    )
    create_table_query = f"CREATE TABLE {table_name} (\n{columns}\n);"
    conn.execute(create_table_query)

    insert_values = ", ".join(
        [f"({', '.join([f'{repr(val)}' for val in row.values])})" for _, row in df.iterrows()]
    )
    insert_query = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES {insert_values};"
    conn.execute(insert_query)
    return None


def generate_paraphrases(original_sentence, num_paraphrases=5, temperature=0):
    """Generates paraphrases for the input sentence using OpenAI's GPT-3 API.

    Parameters
    ----------
    original_sentence : str
        input sentence for which paraphrases are to be generated
    num_paraphrases : int, optional
        number of paraphrases to be generated, by default 5
    temperature : int, optional
        temperature parameter for the GPT-3 API, by default 0

    Returns
    -------
    list
        list of paraphrases
    """
    prompt = f"""Generate {num_paraphrases} different paraphrased sentences split by ';'
                 for the following sentence: '{original_sentence}'\n"""

    model_engine = "text-davinci-003"
    temperature = temperature
    max_tokens = 1000
    stop_sequence = None

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequence,
    )
    paraphrased_sentence = response.choices[0].text.strip()

    paraphrased_sentence = paraphrased_sentence.split(";")
    paraphrased_sentence = [p.replace("\n", "").strip() for p in paraphrased_sentence]
    return paraphrased_sentence


def convert_df_to_csv_string(input_df: pd.DataFrame) -> str:
    """Converts input dataframe to csv as a string variable.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe

    Returns
    -------
    str
        csv string variable
    """
    table_string = input_df.to_csv(index=False)
    return table_string


def get_gpt_token_count(input_data: str, model: str):
    """Returns the number of tokens in the input data.

    Parameters
    ----------
    input_data : str
        Input data
    model : str
        Model name

    Returns
    -------
    int
        Number of tokens in the input data
    """
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(input_data))
    return num_tokens


def download_spacy_nltk_data() -> bool:
    """Function to download spacy and nltk resources automatically.
    Returns
    -------
    bool
        True, if there is an error in downloading nltk data. False, otherwise.
    """
    # TODO: fsspec: Integrate fsspec in nltk data find? Or it can read from local storage?
    error_flag = False
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        print(f"Error:{e}")
        download_flag = nltk.download("punkt", quiet=True)
        if not download_flag:
            error_flag = True

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError as e:
        print(f"Error:{e}")
        download_flag = nltk.download("stopwords", quiet=True)
        if not download_flag:
            error_flag = True

    # Check for spacy en_core_web_lg download
    model_name = "en_core_web_lg"
    available_flag = spacy.util.is_package(model_name)
    if not available_flag:
        try:
            spacy.cli.download(model_name)
        except SystemExit as e:
            print(f"Error:{e}")
            error_flag = True
        except Exception as e:
            print(f"Error:{e}")
            error_flag = True

    return error_flag


def rate_limit_error_handler(
    time_delay: int,
    max_retries: int,
    logger: logging.Logger = None,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry request after sleeping for x seconds.

    Parameters
    ----------
    logger : logging.Logger
        Logger object passed from QueryInsights.
    time_delay : int
        time in seconds to sleep.
    max_retries : int
        Maximum number of retries whenever we face Ratelimiterror
    errors : tuple, optional
        tuple of openAI errors, by default (openai.error.RateLimitError,)

    Returns
    -------
    function
        decorator function

    Raises
    ------
    Exception
        Maximum number of retries exceeded.
    """
    logger = logging.getLogger(MYLOGGERNAME)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors:
                    # Increment retries
                    num_retries += 1
                    logger.info(f"Trial number: {num_retries} failed.")

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        logger.error(f"Maximum number of retries ({max_retries}) exceeded.")
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Sleep for the delay
                    logger.info(
                        f"Request will be retried after sleeping for {time_delay} seconds."
                    )
                    time.sleep(time_delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


def read_text_file(filename, fs=None):
    """Reads the text file and returns the contents.

    Parameters
    ----------
    filename : str
        Path to the text file
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    str
        Contents of the text file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(filename, "r") as f:
        contents = f.read()
    return contents


def convert_data_dictionary_to_pandas_df(data_dictionary: dict):
    """
    Converts the raw data dictionary to a pandas dataframe.

    Parameters
    ----------
    data_dictionary : dict
        Raw data dictionary with table_name and columns as Keys and Description/ID as values

    Returns
    -------
    data_dictionary_df
        Pandas dataframe
    """
    data_dictionary_df = pd.DataFrame()
    for key in data_dictionary.keys():
        dd_table = data_dictionary[key]["columns"]
        df = pd.DataFrame(dd_table)
        df["table_name"] = key
        data_dictionary_df = pd.concat([data_dictionary_df, df], axis=0)

    return data_dictionary_df


def capture_stdout_to_var(func: Callable[[str], None], kwargs: dict) -> Tuple[str, str]:
    """Captures stdout and stderr when we run code blocks to two variables and returns them.

    Parameters
    ----------
    func : Callable[[str], None]
        Function for which we are capturing the stdout and stderr
    kwargs : dict
        keyword arguments for the function.

    Returns
    -------
    Tuple[str, str]
        stdout and stderr.
    """
    f = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(f), redirect_stderr(err):
        func(**kwargs)

    # Get the stdout that gets populated when we run `func` into var `f`.
    out = f.getvalue()
    err_out = err.getvalue()

    return out, err_out


def save_results_as_json(
    question: str,
    predicted_answer: str = "",
    file_save_path: str = None,
    fs=None,
    **kwargs,
) -> None:
    """Saves Track 3 results in JSON format as below::

        {"id": id, "input": {"text": question}, "output": {"text": predicted_answer}}

    Parameters
    ----------
    question : str
        User question
    predicted_answer : str, optional
        Track 3 result, by default ""
    file_save_path : str, optional
        JSON file save path, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Raises
    ------
    ValueError
        if any of the argument is missing.
    """
    if file_save_path is None:
        raise ValueError("Save path of json must be given.")

    fs = fs or fsspec.filesystem("file")
    id_ = "1"

    result = {"id": id_, "input": {"text": question}, "output": {"text": predicted_answer}}

    with fs.open(file_save_path, "w") as f:
        json.dump(result, f, **kwargs)

    return


def multiple_json_to_jsonl(json_folder_path: str, jsonl_file_path: str = None, fs=None):
    """Reads all JSON files in given folder and joins them together in a JSONL file.

    Parameters
    ----------
    json_folder_path : str
        Folder which will be traversed to search for all json files.
    jsonl_file_path : str, optional
        save path for resulting jsonl, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    if jsonl_file_path is None:
        jsonl_file_path = pp.join(json_folder_path, "all_user_query.jsonl")

    all_query_data = []
    fs = fs or fsspec.filesystem("file")
    # Get all json saved already in the path.
    for dirpath, _, files in fs.walk(json_folder_path):
        for file in files:
            if file.endswith(".json"):
                print("Json file found at", pp.join(dirpath, file))
                with fs.open(pp.join(dirpath, file), "r") as json_file:
                    json_data = json.load(json_file)
                    all_query_data.append(json_data)

    # Save these multiple json to jsonl
    with fs.open(jsonl_file_path, "w") as fout:
        for each_json_data in all_query_data:
            json.dump(each_json_data, fout)
            fout.write("\n")
    print("jsonl saved at ", jsonl_file_path)

    return


def format_dataframe(df):
    """
    converts data frame into a front end usable format

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be sent to front end

    Returns
    -------
    list
    converted rows from a dataframe into a list
    """

    # if the index is just the pandas default index, it can be dropped. Else, it will be added to the data frame as a column
    df = df.reset_index()
    if "index" in df.columns:
        del df["index"]

    return df.to_json(orient="records")


def get_word_chunks(doc, stop_words):
    """
    Extracts chunks of words from a given document 'doc' excluding stop words.
    Example:
    For document input of this sentence -
        How much quantity of FlavorCrave brand was shipped each week?
    Result - ['week', 'shipped', 'How much quantity', 'FlavorCrave brand']

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The input document from which chunks will be extracted.
    stop_words : list
        A list of stop words to be excluded from the chunks.

    Returns
    -------
    list
        A list of chunks containing nouns, verbs, proper nouns, adjectives, and adverbs,
        excluding stop words and punctuation.
    """
    # doc = nlp(question)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = [n for n in noun_chunks if n not in stop_words]

    allowed_postags = ["NOUN", "VERB", "PROPN", "ADJ", "ADV"]
    tokens = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in allowed_postags
    ]
    all_chunks = list(set(noun_chunks + tokens))

    result = []
    for word in all_chunks:
        is_subset = any(word in other_word and word != other_word for other_word in all_chunks)
        if not is_subset:
            result.append(word)
    return result


class CustomExceptionHook:
    """
    This class redirects exceptions to predefined functions for both python and jupyter.
    The respective functions in the class is invoked in case of an error and the error message is logged.
    """

    def __init__(self, logger: logging.Logger):
        """
        Parameters
        ----------
        logger : logging.Logger
            Logger object passed from QueryInsights.
        """
        self.logger = logger

    def custom_excepthook(self, exc_type: type, exc_value, exc_traceback: traceback):
        """
        This function is invoked in case of exceptions when code is run in python shell
        Logs given errors to logger

        Parameters
        ----------
        exc_type : type
            Type of error
        exc_value : class
            Error class and message
        exc_traceback : traceback
            Traceback object with the error information
        """
        tb_formatted = traceback.format_exception(exc_type, exc_value, exc_traceback)
        self.logger.error(
            "The following uncaught error occurred" + "\n" + "".join(tb_formatted).strip()
        )

    def custom_excepthook_jupyter(self, shell, exc_type, exc_value, exc_traceback, tb_offset=None):
        """
        This function is invoked in case of exceptions when code is run in jupyter notebooks
        Logs given errors to logger

        Parameters
        ----------
        shell : IPython.shell type
            Ipython shell object
        exc_type : type
            Type of error
        exc_value : class
            Error class and message
        exc_traceback : traceback
            Traceback object with the error information
        tb_offset : IPython object
            IPython offset object
        """
        tb_formatted = traceback.format_exception(exc_type, exc_value, exc_traceback)
        self.logger.error(
            "The following uncaught error occurred" + "\n" + "".join(tb_formatted).strip()
        )


def log_uncaught_errors(logger: logging.Logger):
    """
    Catches uncaught errors in code and logs to logger before exiting

    Parameters
    ----------
    logger : logging.Logger
        Logger object passed from QueryInsights.
    """
    customExceptionHook = CustomExceptionHook(logger)

    if is_jupyter():
        with suppress(Exception):
            # This will error out. Only added to avoid import error message in vscode
            from IPython import get_ipython

        get_ipython().set_custom_exc(
            (Exception,),
            customExceptionHook.custom_excepthook_jupyter,
        )
    else:
        sys.excepthook = customExceptionHook.custom_excepthook


def is_jupyter() -> bool:
    """Check if code is running in jupyter notebook or python interpreter

    Returns
    -------
    bool
        True if code is running in jupyter, False otherwise
    """
    try:
        with suppress(Exception):
            # This will error out. Only added to avoid import error message in vscode
            from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def generate_env_dict(
    cloud_storage_dict: DotifyDict = None, account_key: str = None
) -> Union[dict, None]:
    """
    From given cloud storage parameters, generate an environment dictionary which can be passed to subprocess.run as environment variables

    Parameters
    ----------
    cloud_storage_dict : DotifyDict, optional
        Dictionary containing cloud connection parameters, by default None
    account_key : str, optional
        Account key of the cloud storage provider, by default None

    Returns
    -------
    Union[dict, None]
        If cloud_storage_dict isn't None, return a dictionary containing environment variables, else return None
    """
    # Cloud storage paramters are unpacked into a dictionary
    if cloud_storage_dict.platform is not None:
        env = {**cloud_storage_dict}
        # Removing keys that are not required
        env.pop("account_key_path", None)
        env.pop("_ipython_canary_method_should_not_exist_", None)
        env.pop("platform", None)
        env["AccountKey"] = "" if account_key is None else account_key
        # Below env variables are only required in Windows os for subprocess.run to work
        env["SYSTEMROOT"] = os.getenv("SYSTEMROOT", "")
        env["APPDATA"] = os.getenv("APPDATA", "")
    else:
        env = None

    return env
