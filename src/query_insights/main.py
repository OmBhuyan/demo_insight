import datetime
import logging
import posixpath as pp
import random
import re
import traceback
from typing import Tuple

import numpy as np
import openai
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from sql_metadata import Parser

from .chart_generator import GenerateCharts
from .config_validation import (
    DataConfigValidator,
    ModelConfigValidator,
    UserConfigValidator,
)
from .insights_generator import GenerateInsights
from .pre_processing import DatabaseSetup, HybridQuestionClassifier
from .text_to_query_generator import BotResponse, TextToQuery
from .utils import (
    SensitiveContentError,
    TimeoutError,
    TokenLimitError,
    convert_data_dictionary_to_pandas_df,
    create_logger,
    download_spacy_nltk_data,
    format_dataframe,
    fs_connection,
    get_fs_and_abs_path,
    get_word_chunks,
    load_config,
    load_data_dictionary,
    load_key_to_env,
    log_uncaught_errors,
    read_data,
    read_text_file,
)

MYLOGGERNAME = "QueryInsights"


class QueryInsights:
    """Main class that controls individual tracks outputs. Given a user query, this class provides charts and insights to best answer the given query.

    Parameters
    ----------
    user_config_path : str
        path for input user_config dictionary for storing and accessing user-specific configurations.
    data_config_path : str
        path for input data_config dictionary contains the paths to the data.
    model_config_path : str
        path for input model_config dictionary for storing and accessing model-related configurations.
    api_key : str, optional
        API key string. If left as blank, it will look for the path in the data_config and read the key from there, by default None
    fs_key : str, optional
        Account key for connecting to the File storage. If left as blank and platform specified in the data_config (data_config.cloud_storage.platform) is not blank, it will look for the path in the data_config and read the key from there, by default None
    logging_level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"
    log_file_path : str, optional
        File path to save the logs, by default None
    verbose : bool, optional
        If `True` logs will be printed to console, by default True

    Raises
    ------
    ValueError
        if any of the argument is missing or invalid.

    """

    def __init__(
        self,
        user_config_path: str,
        data_config_path: str,
        model_config_path: str,
        debug_config_path: str,
        api_key: str = None,
        fs_key: str = None,
        logging_level: str = "INFO",
        log_file_path: str = None,
        verbose: bool = True,
    ) -> None:
        """Class constructor"""
        if user_config_path is None:
            raise ValueError("user_config path is a mandatory argument.")
        # load user config
        self.user_config = load_config(cfg_file=user_config_path)
        self.user_config_path = user_config_path

        if data_config_path is None:
            raise ValueError("data_config path is a mandatory argument.")
        # load user config
        self.data_config = load_config(cfg_file=data_config_path)
        self.data_config_path = data_config_path

        if model_config_path is None:
            raise ValueError("model_config path is a mandatory argument.")
        # load user config
        self.model_config = load_config(cfg_file=model_config_path)
        self.model_config_path = model_config_path

        if debug_config_path is None:
            raise ValueError("debug_config_path is a mandatory argument.")
        # load user config
        self.debug_config = load_config(cfg_file=debug_config_path)
        self.debug_config_path = debug_config_path

        prefix_url, storage_options = fs_connection(
            fs_connection_dict=self.data_config.cloud_storage, fs_key=fs_key
        )

        # BLOB_ACCOUNT_KEY variable is set here
        self._fs, _ = get_fs_and_abs_path(path=prefix_url, storage_options=storage_options)

        # load openai api key
        if api_key is not None:
            load_key_to_env(
                secret_key=api_key,
                env_var="OPENAI_API_KEY",
                fs=None,
            )
        else:
            load_key_to_env(
                secret_key=self.data_config.path.api_key_location,
                env_var="OPENAI_API_KEY",
                fs=None,
            )

        # self.api_key = api_key
        self.log_file_path = log_file_path
        self.logging_level = logging_level
        self.verbose = verbose
        self.track1_data_dict = None
        self.text_to_query_path = None
        self.query_to_chart_path = None
        self.table_to_insights_path = None

        # Load database
        database_connection = DatabaseSetup(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            fs=self._fs,
        )
        self.conn = database_connection.connection_db()

        tables = self.data_config.path.input_file_names

        # Initialize data dictionary
        self.data_dictionary = {}
        for table in list(tables.keys()):
            self.data_dictionary[table] = load_data_dictionary(
                pp.join(self.data_config.path.data_dictionary_path, f"{table}.json"),
                fs=self._fs,
            )

        # Response variable initialized for creation of response.json
        self.response_json = {}

        validators = [
            (DataConfigValidator, [data_config_path, fs_key]),
            (UserConfigValidator, [user_config_path]),
            (ModelConfigValidator, [model_config_path]),
        ]

        for validator_cls, args in validators:
            config_path = args[0]
            if validator_cls == DataConfigValidator:
                fs_key = args[1]
                validator = validator_cls(config_file_path=config_path, fs_key=fs_key)
            else:
                validator = validator_cls(config_file_path=config_path)
            result = validator.validate_config()
            if not result:
                raise ValueError(
                    f"Config validation failed for {validator} from {config_path}. Result: {result}"
                )

        # Download NLTK Data for preprocessing for questions
        error_flag = download_spacy_nltk_data()
        if error_flag:
            raise ValueError("Failed to download NLTK punkt/stopwords. Please check with IT team.")

        self.units_to_skip = [
            "integer",
            "count",
            "text",
            None,
            "yyyy-ww",
            "na",
            "unit of measure for the quantity",
            "float",
        ]
        return

    def check_similarity_and_get_question_index(
        self, question, questions_dict, path_dict, status_dict
    ) -> str:
        """
        Calculates similarity with other questions using sentence embeddings and cosine similarity.
        And it gets the index for the new question based on the existing question base.

        Two scenarios can happen in this function after identifying similar questions if available -
        1. If the similarity condition is False or KB has no questions, it creates a new index with the prefix and the timestamp.
            <Prefix>_<Timestamp>_1
        2. If the condition is True, then it takes the maximum value from the dir list and increments the secondary index.
            conditon = True
            dir_list = ['Q_20230622142919300_1', 'Q_20230622142919300_2']
            Then the new index which is returned is 'Q_20230622142919300_3'.

        Parameters
        ----------
        question : str
            New question from the user
        questions_dict : dict
            It has the existing questions from the KB as a dictionary.
            Question indexes are Keys and the questions are values.
        path_dict : dict
            Question indexes are Keys and the results path are values.
        status_dict : dict
            Question indexes are Keys and the results status are values.
        Returns
        -------
        str
            Prefix + "_" Primary index + "_" + Secondary Index
        """
        prefix = "Q"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        # Check if the KB has no existing questions.
        if questions_dict is None:
            new_folder_name = prefix + f"_{now}_1"
        else:
            # Looking for similar question if any in the KB using sentence embeddings and cosine similarity.
            model = SentenceTransformer(self.user_config.similarity_check.model)
            # Get the list of questions and indexes from the dictionary.
            folders_list = [i for i in questions_dict.keys()]
            questions_list = [v for v in questions_dict.values()]
            path_list = [p for p in path_dict.values()]
            status_list = [s for s in status_dict.values()]

            # Encode the list of sentences and the standalone sentence
            encoded_list = model.encode(questions_list, convert_to_tensor=True)
            encoded_new_qs = model.encode(question, convert_to_tensor=True)

            # Calculate cosine similarity and the maximum similarity
            cos_similarities = util.pytorch_cos_sim(encoded_new_qs, encoded_list).tolist()[0]
            max_similarity = max(cos_similarities)
            if (
                max_similarity > self.user_config.similarity_check.threshold
            ):  # Threshold for similarity
                # TODO Do this step only if the user wants the similar query to be used. After UI changes, this need to be updated.
                max_index = cos_similarities.index(max_similarity)
                similar_index = folders_list[max_index]
                similar_question = questions_list[max_index]
                similar_question_path = path_list[max_index]
                similar_status = status_list[max_index].strip("][").split(", ")
                similar_dict = {
                    "index": similar_index,
                    "score": max_similarity,
                    "question": similar_question,
                    "path": similar_question_path,
                    "status": similar_status,
                }
                self.similarity = [True, similar_dict]

                similar_indexes = [
                    i for i in folders_list if i.startswith(similar_index.rsplit("_", 1)[0])
                ]
                folder_name = max(similar_indexes)
                n = int(folder_name.split("_")[2]) + 1
                new_folder_name = (
                    folder_name.split("_")[0] + "_" + folder_name.split("_")[1] + "_{}".format(n)
                )
            else:
                new_folder_name = prefix + f"_{now}_1"

        # adding random integer after timestamp to avoid issues with multiple users
        rand = random.randint(0, 10000)
        new_folder_name = f"{new_folder_name}_{rand}"

        return new_folder_name

    def _preprocess(self, question: str = None, additional_context: str = None) -> None:
        """Does pre processing of the data - creates logger, and creates folders for saving the results

        Parameters
        ----------
        question : str, optional
            Question (may or may not contain bot history) to pass to the GPT. Can be the entire conversation either:
             1. Entire conversation passed as a list [[user Q1, bot response 1], [user Q2, bot response 2], [user Q3, bot response 3], ...]
             2. User questions alone as a list with the follow up questions seperated by `;`
             3. Question string alone and there is no conversation
             , by default None
        additional_context : str, optional
            Additional context provided to answer the user question, by default None

        Raises
        ------
        ValueError
            When code is set to UI mode and user query is not provided as a parameter
        """
        # read the question and additional context from user

        if type(question) == list:
            # if the entire conversation is passed as a list [[user1, bot1], [user2, bot2], [user3, bot3], ...]
            self.bot_history = question.copy()
            question = question[0][0]
        elif (type(question) == str) and (";" in question):
            # if all the followup questions are passes as a ";" separated list
            question_list = question.split(";")
            self.bot_history = [[user.strip(), None] for user in question_list]
            question = question_list[0]
        else:
            # else question is just a string and there is no conversation
            self.bot_history = None

        if question is None and not self.user_config.ui:
            first_msg = "User query is not given as input and main.py is called in UI mode, Hence question and additional context from config file will be used."
            self.question = self.user_config.user_inputs.question
            self.additional_context = self.user_config.user_inputs.additional_context
        if question is None and self.user_config.ui:
            raise ValueError("Code is set to UI mode, so user query is mandatory.")
        else:
            first_msg = "User query and/or additional context is given as input."
            self.question = question
            self.additional_context = additional_context

        self.all_tokens = []

        # Creating the multiple charts flag based on the user question
        multiple_charts_indicator = "multiple charts:"
        self.multiple_charts = False
        if multiple_charts_indicator in self.question.lower():
            self.multiple_charts = True
            # Removing the indicator tag from the question for GPT calls.
            self.question = re.sub(
                multiple_charts_indicator, "", self.question, flags=re.IGNORECASE
            )

        # Check if given query is why question.

        classifier = HybridQuestionClassifier(embedding_model="bert")
        threshold = self.user_config.why_question_threshold

        # Check if its why qn or not.
        if not bool(threshold):
            # why_qn_flag will be False when the threshold is: None, 0, "", False.
            # `if (threshold is None) or (threshold.strip() == ""):` could be used but it would've failed when threshold is float
            self.why_qn_flag = False
        else:
            threshold = float(threshold)
            # TODO: Handle edge cases where user question itself contains ';'
            reason_based_questions = classifier.find_reason_based_questions(
                [question.split(";")[-1].strip()], threshold
            )
            for qn, _ in reason_based_questions:
                if qn == question:
                    self.why_qn_flag = True
                else:
                    self.why_qn_flag = False

        # create the output folder structure for saving logs.
        # TODO: Potential place to fix issue#26
        self.exp_folder = pp.join(
            self.data_config.path.output_path, self.data_config.path.exp_name
        )
        self._fs.makedirs(self.exp_folder, exist_ok=True)  # create folder if not already present.
        self.knowledge_base = pp.join(self.exp_folder, "Knowledge_base.xlsx")
        self.bot_training_data = pp.join(self.exp_folder, "bot_training_data.csv")

        # Initialize some Flags in case the Knowledge base file is not created yet.
        # existing_question flag represents if the user question is an exact match of existing questions in KB.
        # existing_status represents the status of Track 1, 2 and 3 from the previous run.
        self.existing_question = False
        self.existing_status = [None, None, None]
        # index_questions_dict is the dictionary with question index from KB as keys and Questions as values.
        # index_path_dict is the dictionary with question index from KB as keys and results path as values.
        # index_status_dict is the dictionary with question index from KB as keys and results status as values.
        index_questions_dict = None
        index_path_dict = None
        index_status_dict = None
        # Simlairity variable is a list with two elements.
        # First element - Flag which represents if the user question is similar to any of the existing question.
        # Second element - dictionary - (Index of the similar existing question, Similarity Score, Similar question, path, and results status) if the first element turns out to be True.
        # If the first element is False, then second element will be None.
        # This will be updated in the check_similarity_and_get_question_index function where similarity is calculated.
        self.similarity = [False, None]

        # Check for question in existing knowledge base if the file is present.
        if self._fs.exists(self.knowledge_base):
            # Get details from existing knowledge base.
            kb_df = read_data(self.knowledge_base, fs=self._fs)
            kb_df.fillna("", inplace=True)
            kb_df["question_w_context"] = kb_df["question"] + " " + kb_df["additional_context"]

            index_questions_dict = kb_df.copy().set_index("index")["question_w_context"].to_dict()
            index_path_dict = kb_df.copy().set_index("index")["results_path"].to_dict()
            index_status_dict = kb_df.copy().set_index("index")["results_status"].to_dict()

            questions = kb_df["question_w_context"].to_list()
            self.indexes_list = kb_df["index"].to_list()
            status = kb_df["results_status"].to_list()
            questions = [re.sub(r"[^\w\s]", "", q).lower().strip() for q in questions]

            user_question = self.question
            if self.additional_context is not None:
                user_question = user_question + " " + self.additional_context
            if hasattr(self, "bot_history") and (self.bot_history is not None):
                history = self.bot_history
                user_question = " ; ".join([q for [q, a] in history])
            # Check if question already exists in knowledge base.
            user_question = re.sub(r"[^\w\s]", "", user_question).lower().strip()
            if user_question in questions:
                self.existing_question = True
                index = questions.index(user_question)
                self.question_index = self.indexes_list[index]
                self.existing_status = status[index][1:-1].split(", ")
                # self.logger.info(
                #     f"User question is same as an existing question with index - {self.question_index}."
                # )

        # print(self.existing_status)
        # get question index based on existing folders.
        if not self.existing_question:
            user_question = self.question
            if self.additional_context is not None:
                user_question = user_question + " " + self.additional_context
            self.question_index = self.check_similarity_and_get_question_index(
                user_question, index_questions_dict, index_path_dict, index_status_dict
            )
            # self.logger.info(
            #     f"Question index generated for this question is {self.question_index}."
            # )
        # self.logger.debug(
        #     f"Existing question: {self.existing_question}, Question index: {self.question_index}, Existing status: {self.existing_status}"
        # )

        if (
            self.existing_question
            and self.user_config.skip_api_call
            and self.user_config.skip_list
        ):
            skip_list = self.user_config.skip_list
            if self.question in skip_list:
                # self.logger.info(
                #     "Question part of 'Skip List'. Model call will not happen if results are present."
                # )
                self.existing_question = True
            else:
                self.existing_question = False
        elif not self.user_config.skip_api_call:
            self.existing_question = False

        try:
            # print(self.question_index)
            self.output_path = pp.join(
                self.data_config.path.output_path,
                self.data_config.path.exp_name,
                self.question_index,
            )

            # Init logger
            current_ts = str(datetime.datetime.now(datetime.timezone.utc)).replace("+00:00", "Z")

            if self.log_file_path is None:
                log_file = pp.join(self.output_path, f"runtime_{current_ts}.log")
                create_logger(
                    logger_name=MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=log_file,
                    verbose=self.verbose,
                    fs=self._fs,
                )
            else:
                create_logger(
                    logger_name=MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=self.log_file_path,
                    verbose=self.verbose,
                    fs=self._fs,
                )

            self.logger = logging.getLogger(MYLOGGERNAME)
            self.logger.info(
                f"The results will be saved in this output folder: {self.question_index} and output path: {self.output_path}"
            )
            # Create subfolder
            self.text_to_query_path = pp.join(self.output_path, "01_text_to_query")
            self.query_to_chart_path = pp.join(self.output_path, "02_query_to_chart")
            self.table_to_insights_path = pp.join(self.output_path, "03_table_to_insights")

            if self.why_qn_flag:
                folders_to_create = [self.table_to_insights_path]
            else:
                folders_to_create = [
                    self.text_to_query_path,
                    self.query_to_chart_path,
                    self.table_to_insights_path,
                ]

            for folder in folders_to_create:
                self.logger.debug(f"Folder - {folder} is created.")
                self._fs.makedirs(folder, exist_ok=True)
        except Exception:
            self.output_path = pp.join(
                self.data_config.path.output_path,
                self.data_config.path.exp_name,
                self.question_index,
            )

            # Init logger

            if self.log_file_path is None:
                log_file = pp.join(self.output_path, "runtime.log")
                create_logger(
                    logger_name=MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=log_file,
                    verbose=self.verbose,
                    fs=self._fs,
                )
            else:
                create_logger(
                    logger_name=MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=self.log_file_path,
                    verbose=self.verbose,
                    fs=self._fs,
                )

            self.logger = logging.getLogger(MYLOGGERNAME)

            # Create subfolder
            self.text_to_query_path = pp.join(self.output_path, "01_text_to_query")
            self.query_to_chart_path = pp.join(self.output_path, "02_query_to_chart")
            self.table_to_insights_path = pp.join(self.output_path, "03_table_to_insights")

        log_uncaught_errors(self.logger)

        if self.why_qn_flag:
            folders_to_create = [self.table_to_insights_path]
        else:
            folders_to_create = [
                self.text_to_query_path,
                self.query_to_chart_path,
                self.table_to_insights_path,
            ]

            for folder in folders_to_create:
                self.logger.debug(f"Folder - {folder} is created.")
                self._fs.makedirs(folder, exist_ok=True)

        self.logger.info(first_msg)
        self.logger.info(
            f"The results will be saved in this output folder: {self.question_index} and output path: {self.output_path}"
        )
        self.logger.info("Saving Data config")
        self._fs.makedirs(pp.join(self.output_path, "configs"), exist_ok=True)

        # Save data Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "data_config.yaml"), "w"
        ) as fout, open(self.data_config_path, "r") as fin:
            fout.write(fin.read())

        self.logger.info("Saving model config")
        # Save model Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "model_config.yaml"), "w"
        ) as fout, open(self.model_config_path, "r") as fin:
            fout.write(fin.read())

        self.logger.info("Saving user config")
        # Save Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "user_config.yaml"), "w"
        ) as fout, open(self.user_config_path, "r") as fin:
            fout.write(fin.read())

        self.logger.info("Saving debug config")
        # Save Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "debug_config.yaml"), "w"
        ) as fout, open(self.debug_config_path, "r") as fin:
            fout.write(fin.read())

        if self.additional_context is None:
            file_content = f"Question: {self.question}"
            with self._fs.open(pp.join(self.output_path, "question.txt"), "w") as file:
                file.write(file_content)
        else:
            file_content = (
                f"Question: {self.question}\nAdditional context: {self.additional_context}"
            )
            with self._fs.open(pp.join(self.output_path, "question.txt"), "w") as file:
                file.write(file_content)

    def _identify_columns_for_similar_query(self) -> list:
        """
        Identifies columns for a similar query by comparing word chunks and calculating cosine similarities.
        Returns a list of identified columns, the similar question, and the corresponding SQL response.

        Steps followed in the function -
        1. Process the similar question and extract meaningful word chunks.
        2. Process the user question and additional context (if available) and extract meaningful word chunks.
            - Word chunks are identified based on Noun chunks and POS.
        3. Keep only the new word chunks which are part of the user question. Remove the common word chunks b/w similar question and the user question.
        4. Iterate over the new word chunks from the user question and the column descriptions and find the most similar column names using a Sentence Transformer model and cosine similarity.
            - For each new word chunk, columns are identified.
                (The number of such columns identified can be changed from user config file. 10 columns would be ideal.)
        5. Combine these column names with the columns extracted from the SQL query of similar question and return the final list of unique column names.

        Returns
        -------
        list
            [<similar columns list>, similar question, similar response]
        """
        # Getting the columns list from the SQL Query.
        self.logger.info(
            f"Similar question {self.similarity[1]['question']} already exists. Getting the columns from knowledge base."
        )
        # Constructing the path to the SQL query file of the similar question
        similar_qs_path = self.similarity[1]["path"]
        similar_question_path = pp.join(similar_qs_path, "01_text_to_query", "sql_query.sql")
        # Reading the SQL Query from the similar question's folder.
        similar_response = read_text_file(similar_question_path, fs=self._fs)
        # Parsing the SQL query to extract columns
        query_columns = Parser(similar_response).columns

        # Loading the spaCy language model and getting list of stop words.
        nlp = spacy.load("en_core_web_lg")
        stop_words = spacy.lang.en.STOP_WORDS

        # Fetching the similar question and processing it with the language model and extract meaningful word chunks.
        similar_question = self.similarity[1]["question"]
        doc = nlp(similar_question)
        all_chunks_old = get_word_chunks(doc, stop_words)

        # Processing the user question with the language model and extract meaningful word chunks.
        if self.additional_context is None:
            doc = nlp(self.question)
        else:
            doc = nlp(self.question + " " + self.additional_context)
        all_chunks_new = get_word_chunks(doc, stop_words)

        # Filtering only new word chunks present in the user question compared to the old question.
        new_chunks_only = [ch for ch in all_chunks_new if ch not in all_chunks_old]

        # Converting the raw data dictionary to a Pandas DataFrame
        data_dictionary_df = convert_data_dictionary_to_pandas_df(self.data_dictionary)
        data_dictionary_df = data_dictionary_df.reset_index(drop=True)
        # Creating a new column by concatenating table name and column name
        data_dictionary_df["table_column"] = (
            data_dictionary_df["table_name"] + "." + data_dictionary_df["name"]
        )
        # Getting the list of column descriptions and column names.
        description_list = data_dictionary_df["description"].tolist()
        columns_list = data_dictionary_df["name"].tolist()

        # Loading the sentence transformer model and initializing a list to store identified column names.
        model = SentenceTransformer(self.user_config.similarity_check.model)
        possible_columns_list = []
        # Encoding the column descriptions
        encoded_list = model.encode(description_list, convert_to_tensor=True)
        # Looping over the new word chunks from the user question
        for i in range(0, len(new_chunks_only)):
            # Calculating cosine similarities between the new word chunk and column descriptions
            encoded_new_qs = model.encode(new_chunks_only[i], convert_to_tensor=True)
            cos_similarities = util.pytorch_cos_sim(encoded_new_qs, encoded_list).tolist()[0]
            # Getting the indices of the highest similarity values
            max_values = sorted(
                range(len(cos_similarities)), key=lambda x: cos_similarities[x], reverse=True
            )[: self.user_config.similarity_check.num_columns_per_chunk]
            # Getting the corresponding column names and adding it to the list
            max_value_list = [columns_list[j] for j in max_values]
            possible_columns_list = list(set(possible_columns_list + max_value_list))

        final_cols = list(set(possible_columns_list + query_columns))

        return final_cols, similar_question, similar_response

    def text_to_query(self, question: str = None, additional_context: str = None) -> dict:
        """Track 1: Given a user query and dataset, generate SQL and run that SQL on the dataset to get an output dataframe for further processing.

        Parameters
        ----------
        question : str, optional
            Business user query, by default None
        additional_context : str, optional
            Additional context to answer the question, by default None

        Returns
        -------
        dict
            Output format changes depending on track 1 execution resulted in success, failure or skip.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": Tuple[pd.DataFrame, dict],
                }

            If it's failure, format::

                return_value = {
                    "status": "failure",
                    "output": error_message,
                }

            If it's skip, format::

                return_value = {
                    "status": "skip",
                    "output": skip_reason,
                }
        """

        self._preprocess(question=question, additional_context=additional_context)

        # if business_overview exits self variable gets updated
        self.business_overview = None

        if bool(self.data_config.path.business_overview_path):
            if self._fs.exists(self.data_config.path.business_overview_path):
                with self._fs.open(self.data_config.path.business_overview_path, "r") as file_:
                    self.business_overview = file_.read()
                # prompt = (
                #     prompt + "\n\n\nFoloowing is the business overview: \n" + business_overview
                # )

        track1_similarity_chk, similar_response, similar_question = False, None, None
        # Check if there is a similar question present and it's track 1 result is 'success'.
        # TODO: Update this logic using feedback also once it is implemented.
        if self.similarity[0] and self.similarity[1]["status"][0] == "success":
            # Adding 'similar' key to data dictionary with 'Yes' for all the columns
            # which are identified as important for the similar query.
            (
                columns_list,
                similar_question,
                similar_response,
            ) = self._identify_columns_for_similar_query()
            track1_similarity_chk = True
            for table_name in self.data_dictionary.keys():
                columns = self.data_dictionary[table_name]["columns"]
                for column in columns:
                    # For all the columns in the list with format tablename.columnname.
                    if column["name"] in [col for col in columns_list if "." not in col]:
                        column["similar"] = "Yes"
                    # For all the columns in the list with just columnname.
                    if column["name"] in [
                        col.split(".")[1] for col in columns_list if col.startswith(table_name)
                    ]:
                        column["similar"] = "Yes"

        self.logger.info(f"Question to the API: {self.question}")
        self.logger.info(f"Additional Context to the API: {self.additional_context}")

        if self.why_qn_flag:
            skip_reason = f"As given question {self.question} is a why question, user query to SQL generation will be skipped."
            self.logger.info(skip_reason)
            self.track1_output_table = None  # Default
            self.track1_output_table_dict = None  # Default
            return_value = {"status": "skip", "output": (skip_reason, None)}
            return return_value

        # Track 1 - to generate SQL query from the user question
        self.logger.info("SQL generation started.")
        try:
            self.skip_model = False
            if self.existing_question and self.existing_status[0] == "success":
                self.skip_model = True
            if not hasattr(self, "bot_history"):
                self.bot_history = None
            track1_ins = TextToQuery(
                user_config=self.user_config,
                model_config=self.model_config,
                debug_config=self.debug_config,
                question=self.question,
                additional_context=additional_context,
                data_dictionary=self.data_dictionary,
                business_overview=self.business_overview,
                bot_history=self.bot_history,
                db_connection=self.conn,
                output_path=self.text_to_query_path,
                similarity=[track1_similarity_chk, (similar_question, similar_response)],
                skip_model=self.skip_model,
                fs=self._fs,
                # parent_logger=self.logger,
            )
            track1_ins.get_query_suggestion()
            self.logger.info("SQL generation completed.")
            error_message = None
        except TokenLimitError:
            error_message = f"Possible answer for the user query {self.question} exceeded the token limits. Please change the user query or the data."
            self.logger.error(error_message)

        except SensitiveContentError:
            error_message = "The question is flagged as sensitive content by the OpenAI. Please change the language in the question or the data."
            self.logger.error(error_message)

        except TimeoutError:
            error_message = "The request to GPT model timed out (even after retries). Please resubmit the question after sometime or check with your IT team."
            self.logger.error(error_message)

        except openai.error.APIError as e:
            error_message = f"Something went wrong on the OpenAI side. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.Timeout as e:
            error_message = f"Request to GPT timed out. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.RateLimitError as e:
            error_message = (
                f"Ratelimit exceeded. Please resubmit the question after one minute.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.APIConnectionError as e:
            error_message = f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}"
            self.logger.error(error_message)
        except openai.error.AuthenticationError as e:
            error_message = (
                f"The API key may have been expired. Please check with your IT team.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.ServiceUnavailableError as e:
            error_message = f"OpenAI's services are not available at the moment. Please resubmit your question after sometime. If problem still persists, please check with your IT team.\nError:{e}"
            self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error while generating SQL query, error:\n{e}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())

        self.track1_output_query = track1_ins.output_query
        self.track1_output_table_dict = track1_ins.output_table_dict
        self.track1_output_table = track1_ins.output_table

        if hasattr(track1_ins, "query_model_tokens"):
            if track1_ins.query_model_tokens is not None:
                self.all_tokens.append({"Track 1": track1_ins.query_model_tokens.to_dict()})

        self.track1_error_message = error_message

        if error_message is None:
            self.logger.debug(
                f"First 5 rows of the table after running the generated SQL is given below:\n{track1_ins.output_table.head()}"
            )
            self.logger.debug(
                f"Generated table's data dict is given below:\n{track1_ins.output_table_dict}"
            )

            return_value = {
                "status": "success",
                "output": (
                    self.track1_output_query,
                    self.track1_output_table,
                    self.track1_output_table_dict,
                ),
            }
        else:
            return_value = {
                "status": "failure",
                "output": (error_message, self.track1_output_query),
            }

        bot_response = ""
        self.completion_response = ""
        self.completion_error_phrases = [
            "My apologies! Can you ask something else?",
            "Oh no! That didn't work. How about asking a different question?",
            "Sorry, I encountered an error. Maybe try a different query?",
            "Oops, I didn't catch that, can you please modify the question or change it",
            "Looks like I'm having trouble with that one. Can you ask me something else?",
            "Sorry about that, can you try a different question for me?",
            "Hm, I'm not quite sure about that. Can you ask me something else?",
        ]

        if bool(self.user_config.bot_response):
            if return_value["status"] == "failure":
                if self.user_config.bot_response == "rule_based":
                    # to generate the bot response using hardcoded custom responses
                    bot_response = BotResponse(mode="rule_based").get_bot_error_message(
                        error_message
                    )
                elif self.user_config.bot_response == "model_based":
                    # To use davinci 003 for bot response (currently not in use)
                    bot_response_ins = BotResponse(
                        user_config=self.user_config,
                        model_config=self.model_config,
                        conversation_history=self.bot_history,
                        error_message=error_message,
                        skip_model=False,
                        mode="model_based",
                    )
                    bot_response_ins.process_sql_error()
                    bot_response = bot_response_ins.bot_response_output

                if bot_response is None:
                    bot_response = random.choice(self.completion_error_phrases)
                self.completion_response = random.choice(self.completion_error_phrases)

        # Creation of Response JSON for track 1

        question = self.question
        type_ = "insights"


        data_dict = {}
        data_dict["insight_type"] = "sql_query"
        if return_value["status"] == "success":
            data_dict["content"] = return_value["output"][0]
            data_dict["error"] = ""
            data_dict["showError"] = False
        elif return_value["status"] == "failure":
            data_dict["content"] = return_value["output"][1]
            data_dict["error"] = return_value["output"][0]
            data_dict["showError"] = True
        data_dict["bot_response"] = bot_response

        request_json = {}
        response_json = {}

        request_json["question"] = question
        response_json["error"] = ""
        response_json["status"] = [return_value["status"]]
        response_json["type"] = type_
        response_json["data"] = [data_dict]
        response_json["created_time"] = str(datetime.datetime.now(datetime.timezone.utc)).replace("+00:00", "Z")
        response_json["completion_response"] = self.completion_response
        response_json["response_for_history"] = bot_response

        self.response_json["Request JSON"] = request_json
        self.response_json["Response JSON"] = response_json

        # JSON is created to be used for front-end applications.

        return return_value

    def _check_empty_charts(self, chart_object) -> bool:
        """
        Function to check whether the chart is empty without axis values or all axis values as 0.

        Parameters
        ----------
        chart_object : JSON
            Plotly Chart Object

        Returns
        -------
        bool
            Boolean to check whether the chart is empty without axis values or all axis values as 0.
        """
        all_zeros = False
        self.logger.info("Checking if all the axis values are 0 in the chart.")
        # Initialize the y-values to an empty list and loop over the chart object for all y-values and append to this list.
        axis_values = []
        for trace in chart_object["data"]:
            # Check if x and y numeric values are present in the chart object.
            # For some plotly charts like Tabular views/histogram, y parameter is not present in the JSON.
            for axis in ["x", "y", "z"]:
                if axis in trace:
                    if isinstance(trace[axis], (np.ndarray, pd.Series)):
                        axis_values = axis_values + trace[axis].tolist()
                    elif isinstance(trace[axis], list):
                        axis_values = axis_values + trace[axis]
                    elif isinstance(trace[axis], tuple):
                        axis_values = axis_values + list(trace[axis])
                    else:
                        self.logger.info("Chart object has other data type.")

        # Remove any non-numeric values from the list since we are looking at both x and y axis.
        axis_values = [v for v in axis_values if isinstance(v, (int, float)) or v is None]
        # Check if all the values are 0 in the y-values list and change the all_zeros flag.
        if len(axis_values) > 0 and all(value == 0 or value is None for value in axis_values):
            all_zeros = True
            self.logger.info("All axis-values are 0 in the chart. So the chart can be skipped.")
        else:
            if len(axis_values) == 0:
                self.logger.info("No axis-values present. It could be a tabular view.")
            else:
                self.logger.info("Axis-values other than 0 present in the chart object.")

        return all_zeros

    def query_to_chart(
        self,
        question: str = None,
        additional_context: str = None,
        track1_output_table: pd.DataFrame = None,
        track1_output_table_dict: dict = None,
    ) -> dict:
        """Track 2: Given user query, generate python code that generates chart and display to the user.

        Parameters
        ----------
        question : str, optional
            Business user query, by default None
        additional_context : str, optional
            Additional context to answer the question, by default None
        track1_output_table : pd.DataFrame, optional
            Output of text_to_query function by running the SQL generated to answer the ``question``, by default None
        track1_output_table_dict : dict, optional
            Data dictionary of ``track1_output_table`` parameter, by default None

        Returns
        -------
        dict
            Output format changes depending on track 1 execution resulted in success, skip or failure.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": (chart object, track 1 table),
                }

            If chart object is None, format::

                return_value = {
                    "status": "skip",
                    "output": (None, track 1 table),
                }

            If track 2 have an error, format::

                return_value = {
                    "status": "failure",
                    "output": (error message, track 1 table),
                }
        """
        if question is None:
            question = self.question

        if additional_context is None:
            additional_context = self.additional_context

        # Check if its a why Question.
        if self.why_qn_flag:
            skip_reason = f"As given question {question} is a why question, chart generation will be skipped."
            self.logger.info(skip_reason)
            return_value = {
                "status": "skip",
                "output": skip_reason,
            }
            return return_value

        # Input data validation
        if track1_output_table is None:
            track1_output_table = self.track1_output_table

        if track1_output_table_dict is None:
            track1_output_table_dict = self.track1_output_table_dict

        # Track 2 - to generate chart type/code suggestion from Track 1 results
        # track1_output_table_dict = track1_output_table_dict.replace(

        # )  # replace single quotes with double quotes
        # track1_data_dict = {"columns": json.loads(track1_output_table_dict)}
        alternate_dict = None
        if track1_output_table is not None:
            alternate_dict = {
                "columns": {col: dtype.name for col, dtype in track1_output_table.dtypes.items()}
            }

        # Initializing the track 1 data dict to None.
        track1_data_dict = None
        try:
            self.logger.info("Reading the Track 1's data dictionary.")
            # track1_data_dict = ast.literal_eval("{'columns':" + track1_output_table_dict + "}")
            # Create a new dictionary with the "columns" key and the list of dictionaries as its value
            # dict_list = _string_to_dict(track1_output_table_dict)
            dict_list = track1_output_table_dict
            if len(dict_list) == track1_output_table.shape[1]:
                track1_data_dict = {"columns": dict_list}
                self.logger.info(f"Track 1 data dictionary is read - {track1_data_dict}")
            else:
                self.logger.info(
                    """Some issues with data dictionary. It doesn't have all the column details of track 1 result.
                    Changing that to columns list."""
                )
                track1_data_dict = alternate_dict
        except Exception as e:
            self.logger.info(
                f"Some error with the Track 1's data dictionary. Changing that to columns list. Error - {e}"
            )
            track1_data_dict = alternate_dict

        # print(track1_data_dict)

        self.logger.info("User query to chart generation started.")
        self.chart_object = None  # Default
        try:
            if self.multiple_charts:
                self.logger.info(
                    "User has opted for more than one chart. Track 2 will be processed accordingly."
                )
            self.skip_model = False
            if self.existing_question and self.existing_status[1] == "success":
                self.skip_model = True
            track2_ins = GenerateCharts(
                user_config=self.user_config,
                data_config=self.data_config,
                model_config=self.model_config,
                question=question,
                additional_context=additional_context,
                table=track1_output_table,
                data_dictionary=track1_data_dict,
                business_overview=self.business_overview,
                output_path=self.query_to_chart_path,
                skip_model=self.skip_model,
                sql_results_path=self.text_to_query_path,
                multiple_charts=self.multiple_charts,
                fs=self._fs,
                # parent_logger=self.logger,
            )
            self.chart_object = track2_ins.process_suggestion()
            self.logger.info("User query to chart generation completed.")
            error_message = None
        except TokenLimitError:
            error_message = f"Possible answer for the user query {self.question} exceeded the token limits. Please change the user query or the data."
            self.logger.error(error_message)

        except SensitiveContentError:
            error_message = "The question is flagged as sensitive content by the OpenAI. Please change the language in the question or the data."
            self.logger.error(error_message)

        except TimeoutError:
            error_message = "The request to GPT mdoel timed out (even after retries). Please resubmit the question after sometime or check with your IT team."
            self.logger.error(error_message)

        except openai.error.APIError as e:
            error_message = f"Something went wrong on the OpenAI side. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.Timeout as e:
            error_message = f"Request to GPT timed out. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.RateLimitError as e:
            error_message = (
                f"Ratelimit exceeded. Please resubmit the question after one minute.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.APIConnectionError as e:
            error_message = f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}"
            self.logger.error(error_message)
        except openai.error.AuthenticationError as e:
            error_message = (
                f"The API key may have been expired. Please check with your IT team.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.ServiceUnavailableError as e:
            error_message = f"OpenAI's services are not available at the moment. Please resubmit your question after sometime. If problem still persists, please check with your IT team.\nError:{e}"
            self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error while generating the chart, error:\n{e}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())

        # Check if all the axis values are 0.
        # Initializing the all_zeros flag to False.
        if self.chart_object is not None:
            self.all_zeros = [False] * len(self.chart_object)
            for i in range(0, len(self.chart_object)):
                if self.chart_object[i] is not None:
                    try:
                        if "Chart Metrics" in self.chart_object[i]:
                            # Since the metrics can have 0 in it, we can keep the metrics as is.
                            # So keeping the Flag as False.
                            self.all_zeros[i] = False
                        else:
                            self.all_zeros[i] = self._check_empty_charts(self.chart_object[i])
                    except Exception as e:
                        error_string = f"Error in identifying empty chart for {i}. \n Error: {e}"
                        self.logger.info(error_string)

        if hasattr(track2_ins, "chart_type_tokens"):
            if track2_ins.chart_type_tokens is not None:
                self.all_tokens.append({"Track 2a": track2_ins.chart_type_tokens.to_dict()})
        if hasattr(track2_ins, "chart_code_tokens"):
            if track2_ins.chart_code_tokens is not None:
                self.all_tokens.append({"Track 2b": track2_ins.chart_code_tokens.to_dict()})

        if error_message is None:
            if self.chart_object is None or all(self.all_zeros):
                if isinstance(self.chart_object, list):
                    if len(self.chart_object) == 1:
                        self.chart_object = self.chart_object[0]
                return_value = {
                    "status": "skip",
                    "output": (self.chart_object, track1_output_table),
                }
            else:
                # Remove the list if there is only one element in chart object.
                if len(self.chart_object) == 1:
                    self.chart_object = self.chart_object[0]
                return_value = {
                    "status": "success",
                    "output": (self.chart_object, track1_output_table),
                }
        else:
            return_value = {
                "status": "failure",
                "output": (error_message, track1_output_table),
            }

        # Creation of Response JSON for track 2
        self.response_json["Response JSON"]["status"].append(return_value["status"])

        data_dict = {}
        if return_value["status"] == "success":
            # when track 2 runs successfully
            data_dict["insight_type"] = "chart"
            data_dict["content"] = self.chart_object.to_dict()
            data_dict["error"] = ""
            data_dict["showError"] = False

        else:
            # when track 2 has failed or been skipped
            data_dict["insight_type"] = "table"
            table = return_value["output"][1]
            if table.shape == (1, 1):
                # if track2 is skipped and returns a scalar value in a 1x1 table
                unit = self._find_units_of_measurement()
                data_dict["insight_type"] = "scalar"
                data_dict["content"] = f"{table.values[0][0]} {unit}"
            else:
                # if track2 is skipped and returns table
                data_dict["insight_type"] = "table"
                data_dict["content"] = [format_dataframe(table)]

            if return_value["status"] == "failure":
                # show error message when track 2 fails
                data_dict["error"] = return_value["output"][0]
                data_dict["showError"] = True
            else:
                # show error message when track 2 gets skipped
                data_dict["error"] = ""
                data_dict["showError"] = False
        data_dict["bot_response"] = ""


        if return_value["status"] == "failure":
            self.completion_response = random.choice(self.completion_error_phrases)

        self.response_json["Response JSON"]["data"].append(data_dict)
        # JSON is created to be used for front-end applications.

        return return_value

    def table_to_insights(
        self,
        question: str = None,
        track1_output_table: pd.DataFrame = None,
        track1_output_table_dict: dict = None,
    ) -> dict:
        """Track 3: Given user query, generate python code to derive insights on the underlying table and summarize to a business audience.

        Parameters
        ----------
        question : str, optional
            Business user query, by default None
        track1_output_table : pd.DataFrame, optional
            Output of text_to_query function by running the SQL generated to answer the ``question``, by default None
        track1_output_table_dict : dict, optional
            Data dictionary of ``track1_output_table`` parameter, by default None

        Returns
        -------
        dict
            Output format changes depending on track 3 execution resulted in success or failure.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": actual_output,
                }

            If it's failure, format::

                return_value = {
                    "status": "failure",
                    "output": error_message,
                }

        """
        # Remove the temp directory if persists from prev run for the same qn
        if self._fs.exists(pp.join(self.table_to_insights_path, "tmp")):
            self._fs.rm(pp.join(self.table_to_insights_path, "tmp"), recursive=True)

        if question is None:
            question = self.question

        # check if its a why question.
        if self.why_qn_flag:
            self.logger.info(
                f"As given question {question} is a why question, track 3 will run on entire data and data dictionary."
            )
            track1_output_table_dict = self.data_dictionary
            track1_output_table = (
                self.track1_output_table
            )  # TODO: Probably need to think what table to use after table selector is ready.
        else:  # Else, we have generated track 1 output. Fetch it if its not passed directly as arguments.
            # Input data validation
            if track1_output_table is None:
                track1_output_table = self.track1_output_table

            if track1_output_table_dict is None:
                track1_output_table_dict = self.track1_output_table_dict

        self.logger.info("User query to insight generation started.")
        try:
            self.skip_model = False
            if self.existing_question and self.existing_status[2] == "success":
                self.skip_model = True

            # self.logger.debug("Track 3 skip model - ", self.skip_model)

            gi = GenerateInsights(
                user_config=self.user_config,
                data_config=self.data_config,
                model_config=self.model_config,
                question=question,
                dictionary=track1_output_table_dict,
                business_overview=self.business_overview,
                table=track1_output_table,
                skip_model=self.skip_model,
                output_path=self.table_to_insights_path,
                sql_results_path=self.text_to_query_path,
                fs=self._fs,
            )
            insights = gi.get_insights(units_to_skip=self.units_to_skip)
            self.logger.info("User query to insight generation completed.")
            error_message = None

        except TokenLimitError:
            error_message = f"Possible answer for the user query {self.question} exceeded the token limits. Please change the user query or the data."
            self.logger.error(error_message)

        except SensitiveContentError:
            error_message = "The question is flagged as sensitive content by the OpenAI. Please change the language in the question or the data."
            self.logger.error(error_message)

        except TimeoutError:
            error_message = "The request to GPT mdoel timed out (even after retries). Please resubmit the question after sometime or check with your IT team."
            self.logger.error(error_message)

        except openai.error.APIError as e:
            error_message = f"Something went wrong on the OpenAI side. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.Timeout as e:
            error_message = f"Request to GPT timed out. Please resubmit the question.\nError:{e}"
            self.logger.error(error_message)

        except openai.error.RateLimitError as e:
            error_message = (
                f"Ratelimit exceeded. Please resubmit the question after one minute.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.APIConnectionError as e:
            error_message = f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}"
            self.logger.error(error_message)
        except openai.error.AuthenticationError as e:
            error_message = (
                f"The API key may have been expired. Please check with your IT team.\nError:{e}"
            )
            self.logger.error(error_message)

        except openai.error.ServiceUnavailableError as e:
            error_message = f"OpenAI's services are not available at the moment. Please resubmit your question after sometime. If problem still persists, please check with your IT team.\nError:{e}"
            self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error while generating Insights, error:\n{e}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())

        if hasattr(gi, "question_tokens"):
            if gi.question_tokens is not None:
                self.all_tokens.append({"Track 3a": gi.question_tokens.to_dict()})
        if hasattr(gi, "code_tokens"):
            if gi.code_tokens is not None:
                self.all_tokens.append({"Track 3b": gi.code_tokens.to_dict()})
        if hasattr(gi, "summary_tokens"):
            if gi.summary_tokens is not None:
                self.all_tokens.append({"Track 3c": gi.summary_tokens.to_dict()})

        if error_message is None:
            return_value = {"status": "success", "output": insights}
        else:
            return_value = {"status": "failure", "output": error_message}

        # Remove the temp directory
        if self._fs.exists(pp.join(self.table_to_insights_path, "tmp")):
            self._fs.rm(pp.join(self.table_to_insights_path, "tmp"), recursive=True)

        # Creation of Response JSON for track 3
        self.response_json["Response JSON"]["status"].append(return_value["status"])


        data_dict = {}
        data_dict["insight_type"] = "summary"
        if return_value["status"] == "success":
            data_dict["content"] = return_value["output"]
            data_dict["error"] = ""
            data_dict["showError"] = False
        elif return_value["status"] == "failure":
            data_dict["content"] = ""
            data_dict["error"] = return_value["output"]
            data_dict["showError"] = True
        data_dict["bot_response"] = ""


        self.response_json["Response JSON"]["data"].append(data_dict)
        # JSON is created to be used for front-end applications.

        completion_success_phrases = [
            "Done! Here are the results.",
            "All set! Here's what I found.",
            "Got it! Here are the results for you.",
            "Finished! Here's what you were looking for.",
            "Here's what I came up with.",
            "Completed! Here's what I found out for you.",
            "Task accomplished! Here are the results you requested.",
            "Task fulfilled! Here's what I found.",
            "Done and done! Here are the results you asked for.",
        ]

        status = self.response_json["Response JSON"]["status"]
        # if track 1 fails, response_for_history is updated in text to query function with bot response
        # if track 2 fails, table is returned so response can still be success message
        if status[2]=="failure":
            # if track 3 fails, response_for_history is updated with ""
            self.completion_response = random.choice(self.completion_error_phrases)
            self.response_json["Response JSON"]["response_for_history"] = ""
        else:
            # if none of the tracks fail, response_for_history is updated with success message
            self.completion_response = random.choice(completion_success_phrases)
            self.response_json["Response JSON"]["response_for_history"] = self.completion_response
        self.response_json["Response JSON"]["completion_response"] = self.completion_response

        return return_value

    def _find_units_of_measurement(self) -> str:
        """Helper method to extract units of measurement from Track 1 output data dict.

        Returns
        -------
        str
            units of measurement
        """
        if (not hasattr(self, "track1_output_table_dict")) | (
            self.track1_output_table_dict is None
        ):
            self.logger.error("Track 1 is not called or errored out. Thus there is no units.")
            unit = ""
        else:
            # data_dict = literal_eval(self.track1_output_table_dict)
            # data_dict = _string_to_dict(self.track1_output_table_dict)
            data_dict = self.track1_output_table_dict
            if len(data_dict) != 1:
                self.logger.info(
                    "Track 1 output is not scalar, thus we cannot use units of measurement."
                )
                unit = ""
            else:
                if "unit_of_measurement" not in list(data_dict[0].keys()):
                    unit = ""
                    self.logger.info("The data dictionary doesn't have units of measurement")
                else:
                    unit = data_dict[0]["unit_of_measurement"]
                    if unit.lower() in self.units_to_skip:
                        unit = ""
                    self.logger.info(f"The units of measurement is {unit}.")

        return unit

    def create_bot_training_data(self, history):
        """
        creates and saves a training data that contains the user questions, chat conversation, generated query, error message (if any) and the bot's output response

        Parameters
        ----------
        history : list
            contains the current entire history of the chat conversation. example: [[user_msg1, bot_response1],[user_msg2, bot_response2],[user_msg3,bot_response2],...]
        """
        if bool(history):
            training_data = {}
            training_data["question_index"] = self.question_index
            training_data["original_question"] = self.question
            training_data["conversation"] = ""
            for conv in history[:-1]:
                training_data["conversation"] += f"user: {conv[0]}\n"
                training_data["conversation"] += f"bot: {conv[1]}\n"
            training_data["conversation"] += f"user: {history[-1][0]}"

            training_data["query"] = self.track1_output_query

            training_data["error_message"] = self.track1_error_message
            training_data["bot_response"] = history[-1][1]
            training_data = pd.DataFrame(training_data, index=[0])
            if self._fs.exists(self.bot_training_data):
                # TODO: fsspec: Test if this works
                with self._fs.open(self.bot_training_data, mode="rb") as fp:
                    training_data_all = pd.read_csv(fp)
                training_data = pd.concat([training_data_all, training_data])
                with self._fs.open(self.bot_training_data, mode="wb", newline="") as fp:
                    training_data.to_csv(fp, index=False)
            else:
                with self._fs.open(self.bot_training_data, mode="wb", newline="") as fp:
                    training_data.to_csv(fp, index=False)

    def update_knowledgebase(self, alltracks_status, feedback) -> None:
        """
        Creates the Knowledge base excel if it's not already available with the results.
        If it's available, it will add another row to the existing excel with the results.
        The new/updated row will include the below details -
        1. Question index
        2. Question/Context
        3. Results Status
        4. Results path

        Parameters
        ----------
        alltracks_status : list
            contains list all 3 tracks status. Example ['success', 'skip', 'success']
        feedback : list
            feedback from UI

        Returns
        -------
        None
        """
        try:
            # update the knowledge base
            self.logger.info("Updating the Knowledge base...")
            alltracks_status_str = "[" + ", ".join(alltracks_status) + "]"
            if len(self.all_tokens) > 0:
                all_tokens_str = "[" + ", ".join([str(d) for d in self.all_tokens]) + "]"
            else:
                all_tokens_str = ""

            empty_chart_flag = 0
            if hasattr(self, "all_zeros"):
                if all(self.all_zeros):
                    empty_chart_flag = 1

            total_tokens = ""
            if len(self.all_tokens) > 0:
                total_tokens = {
                    "Total": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
                }

                for item in self.all_tokens:
                    for key, values in item.items():
                        for k, v in values.items():
                            total_tokens["Total"][k] += v

                total_tokens = str(total_tokens["Total"])

            if self.similarity[0]:
                similar_question = self.similarity[1]["question"]
                similarity_score = self.similarity[1]["score"]
            else:
                similar_question, similarity_score = None, None

            col_list = [
                "index",
                "question",
                "additional_context",
                "results_status",
                "results_path",
                "feedback",
                "similarity_score",
                "similar_question",
                "lastmodifiedtime",
                "unit_of_measurement",
                "empty_chart_flag",
                "token_information",
                "total_tokens",
            ]

            question = self.question
            if hasattr(self, "bot_history") and (self.bot_history is not None):
                history = self.bot_history
                question = " ; ".join([q for [q, a] in history])

            unit = self._find_units_of_measurement()
            val_list = [
                self.question_index,
                question,
                self.additional_context,
                alltracks_status_str,
                self.output_path,
                feedback,
                similarity_score,
                similar_question,
                str(datetime.datetime.now(datetime.timezone.utc)).replace("+00:00", "Z"),
                unit,
                empty_chart_flag,
                all_tokens_str,
                total_tokens,
            ]

            if self._fs.exists(self.knowledge_base):
                kb_df = read_data(self.knowledge_base, fs=self._fs)
                self.logger.info("Deleting existing row(s) with same index to avoid duplicates.")
                kb_df = kb_df[~(kb_df["index"] == self.question_index)]

                update_data = pd.DataFrame(
                    dict(zip(col_list, val_list)), index=[kb_df.index.max() + 1]
                )
                kb_df = pd.concat([kb_df, update_data])

                with self._fs.open(self.knowledge_base, mode="wb", newline="") as fp:
                    kb_df.to_excel(fp, index=False)
            else:
                update_data = dict(zip(col_list, val_list))
                kb_df = pd.DataFrame([update_data])
                with self._fs.open(self.knowledge_base, mode="wb", newline="") as fp:
                    kb_df.to_excel(fp, index=False)

            self.logger.info("Knowledge base is updated.")

        except Exception as e:
            self.logger.error(f"Error in updating the knowledge base. Error :\n {e}")
            self.logger.error(traceback.format_exc())

    def _skip_tracks(self, get_query_func_response):
        skip_track2 = False
        skip_track3 = False
        if get_query_func_response["status"] == "failure":
            skip_track2 = True
            skip_track3 = True
        if get_query_func_response["status"] == "skip":
            skip_track2 = True
        return skip_track2, skip_track3

    def run_query_insights(
        self, question: str = None, additional_context: str = None
    ) -> Tuple[dict, dict, dict, dict]:
        """For a given question that may have additional context, run all tracks to generate chart and insight summary and display it to the user.
        Parameters
        ----------
        question : str, optional
            Business user query, by default None
        additional_context : str, optional
            Additional context to answer the question, by default None
        Returns
        -------
        Tuple[dict, dict, dict]
            Tuple containing Track 1, 2 and 3 output in any of the below format.
            Output format changes depending on track 2/3 execution resulted in success, failure or skip.
            If it's success, format::
                return_value = {
                    "status": "success",
                    "output": actual_output,
                }
            If it's failure, format::
                return_value = {
                    "status": "failure",
                    "output": error_message,
                }
            If it's skip, format::
                return_value = {
                    "status": "skip",
                    "output": skip_reason,
                }
        """

        # Track 1
        track1_output = self.text_to_query(
            question=question, additional_context=additional_context
        )

        skip_track2, skip_track3 = self._skip_tracks(track1_output)

        # Track 2
        if not skip_track2:
            track2_output = self.query_to_chart()
            if track2_output["status"] == "success":
                self.logger.info("Chart is generated successfully.")
            else:
                self.logger.info(
                    f"No chart is created for this question or there is an error in chart code. Error: {track2_output['output']}"
                )
        else:
            track2_output = {"status": "skip", "output": None}

        # Track 3
        if not skip_track3:
            track3_output = self.table_to_insights()
            # if track3_output["status"] == "failure":
            # raise ValueError(
            #     f"There was an error while running track 3. Error: {track3_output['output']}. Check logs for more information."
            # )
        else:
            track3_output = {"status": "skip", "output": None}

        alltracks_status = [
            track1_output["status"],
            track2_output["status"],
            track3_output["status"],
        ]

        feedback = None
        self.update_knowledgebase(alltracks_status, feedback)

        return track1_output, track2_output, track3_output, self.response_json


# if __name__ == "__main__":
#     pass
#     pass
