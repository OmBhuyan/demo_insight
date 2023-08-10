import logging
import re
from typing import List, Tuple

import fsspec
import gensim.downloader as api
import gensim.models
import mysql.connector
import nltk
import numpy as np

# import pandas as pd
import torch
from mysql.connector import errorcode
from transformers import BertModel, BertTokenizer

from .utils import load_db_credentials, load_sqlite3_database

embedding_model = (
    "BERT"  # BERT/GloVe- Use BERT for better performance,for glove use threshold = 0.6
)
threshold = 0.7  # Threshold for semantic similarity classification

# TODO: Use TigerNLP one after its moved there

MYLOGGERNAME = "QueryInsights"


class HybridQuestionClassifier:
    """Class which helps in identifying 'why' questions.

    Parameters
    ----------
    embedding_model : str, optional
        Embedding model to be used for semantic similarity classification.
        Currently, only "BERT" and "GloVe" are supported, by default "BERT"

    Raises
    ------
    ValueError
        If the given embedding model is not supported.

    Example
    -------
    >>> questions = ["what is the reason for delay in shipment?",
        "why is the shipment delayed?"]
    >>> threshold = 0.7
    >>> classifier = HybridQuestionClassifier(embedding_model="bert")
    >>> reason_based_questions = classifier.find_reason_based_questions(questions, threshold)

    >>> print("Reason based questions:")
    >>> for question, score in reason_based_questions:
            print(f"{question} (score: {score:.2f})")
    """

    def __init__(self, embedding_model: str = embedding_model):
        """Initializes a HybridQuestionClassifier object."""
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        if embedding_model.lower() == "glove":
            self.embedding_model = api.load("glove-wiki-gigaword-100")
        elif embedding_model.lower() == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")

    def preprocess_question(self, question: str) -> List[str]:
        """Preprocesses a given question.

        Parameters
        ----------
        question : str
            Question to be preprocessed.

        Returns
        -------
        List[str]
            List of tokens after preprocessing.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> tokens = classifier.preprocess_question(question)
        >>> print(tokens)
        ['reason', 'delay', 'shipment']
        """
        tokens = nltk.word_tokenize(question.lower())
        tokens = [token for token in tokens if token not in self.stopwords and token.isalnum()]
        return tokens

    def question_vector(self, tokens: List[str]):
        """Creates a vector representation of a given question.

        Parameters
        ----------
        tokens : List[str]
            List of tokens of a question.

        Returns
        -------
        np.ndarray
            Vector representation of the question.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> tokens = ["reason", "delay", "shipment"]
        >>> vector = classifier.question_vector(tokens)
        >>> print(vector)
        [-0.123 0.456 0.789]
        """
        if hasattr(self, "embedding_model"):  # GloVe
            vectors = [
                self.embedding_model[token] for token in tokens if token in self.embedding_model
            ]

            return np.mean(vectors, axis=0)
        elif hasattr(self, "model"):  # BERT
            input_ids = self.tokenizer(
                tokens, return_tensors="pt", padding=True, truncation=True
            ).data["input_ids"]
            with torch.no_grad():
                outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            return embeddings[0]

    def regex_based_classification(self, question: str) -> bool:
        """Classifies a given question as a reason based question using regex.

        Parameters
        ----------
        question : str
            Question to be classified.

        Returns
        -------
        bool
            True if the question is a reason based question, False otherwise.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> is_reason_based = classifier.regex_based_classification(question)
        >>> print(is_reason_based)
        True
        """
        # TODO: move this pattern to config
        pattern = r"""\b(why|reason|reasons|caused|causes|explain|because|purpose|what is the (cause|causing|motivation|rationale|basis
                                |reason|source|origin|root|underlying cause|account|clarification
                                |interpretation|elucidation|description|statement)
                                |what (led to|are the (causes|factors|elements|components|aspects|variables
                                |parameters|determinants|influences))|what is the (explanation
                                |account|clarification|interpretation|elucidation|description
                                |statement|justification|defense|vindication|excuse|warrant
                                |grounds|logic|argument))\b"""
        return bool(re.search(pattern, question, re.IGNORECASE))

    def semantic_similarity_classification(self, question: str) -> float:
        """Classifies a given question as a reason based question using semantic similarity.

        Parameters
        ----------
        question : str
            Question to be classified.

        Returns
        -------
        float
            Similarity score between the question and the reason based question.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> similarity_score = classifier.semantic_similarity_classification(question)
        >>> print(similarity_score)
        0.89
        """
        # TODO: Use TigerNLP one after its moved there
        reason_vector = self.question_vector(self.preprocess_question("why reason"))
        question_vector = self.question_vector(self.preprocess_question(question))

        if question_vector is None:
            return 0

        similarity = gensim.models.KeyedVectors.cosine_similarities(
            reason_vector, question_vector.reshape(1, -1)
        )[0]
        return similarity

    def classify_question(self, question: str, weights: Tuple[float, float] = (0.5, 0.5)) -> float:
        """Classifies a given question as a reason based question using both regex and semantic similarity.

        Parameters
        ----------
        question : str
            Question to be classified.
        weights : Tuple[float, float], optional
            Weights to be used for combining regex and semantic similarity scores, by default (0.5, 0.5)

        Returns
        -------
        float
            Weighted score between the regex and semantic similarity scores.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> weighted_score = classifier.classify_question(question)
        >>> print(weighted_score)
        0.89
        """
        regex_score = self.regex_based_classification(question)
        semantic_similarity_score = self.semantic_similarity_classification(question)
        weighted_score = weights[0] * regex_score + weights[1] * semantic_similarity_score
        return weighted_score

    def find_reason_based_questions(
        self, questions: List[str], threshold: float
    ) -> List[Tuple[str, float]]:
        """Finds reason based questions from a list of questions.

        Parameters
        ----------
        questions : List[str]
            List of questions to be classified.
        threshold : float
            Threshold to be used for filtering the questions.

        Returns
        -------
        List[Tuple[str, float]]
            List of reason based questions with their scores.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> questions = ["What is the reason for delay in shipment?", "What is the reason for delay in shipment?"]
        >>> reason_based_questions = classifier.find_reason_based_questions(questions, threshold=0.5)
        >>> print(reason_based_questions)
        [("What is the reason for delay in shipment?", 0.89), ("What is the reason for delay in shipment?", 0.89)]
        """
        reason_based_questions = []
        for question in questions:
            score = self.classify_question(question)
            if score >= threshold:
                reason_based_questions.append((question, score))
        return sorted(reason_based_questions, key=lambda x: x[1], reverse=True)


class DBConnection:
    """Connecting to database and creating a table in that database

    Parameters
    ----------
    df : Dataframe
        A dataset in the form of a dataframe that needs to be imported into database.
    table_name : str
        Table name for the imported dataframe in database.
    database : str
        Database server example: mysql,sqlite
    data_config : dict
        input data_config dictionary contains the paths to the data.
    database_name : str
        Database name where the table will be stored .
    logging_level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"
    log_file_path : str, optional
        File path to save the logs, by default None
    verbose : bool, optional
        If `True` logs will be printed to console, by default True
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    """

    def __init__(
        self,
        data_config: dict,
        database_name: str = None,
        fs=None,
    ):
        # if parent_logger is not None:
        #     self.logger = parent_logger
        # else:
        #     self.logger = MyLogger(
        #         level=logging_level, log_file_path=log_file_path, verbose=verbose
        #     ).logger
        self.logger = logging.getLogger(MYLOGGERNAME)

        # self.df = df
        # self.table_name = table_name
        self.database = data_config.db_params.db_name
        self.data_config = data_config
        self.database_name = database_name
        self._fs = fs or fsspec.filesystem("file")

        list_of_databases = ["sqlite", "mysql"]

        if self.database not in list_of_databases:
            self.logger.error(
                f"Given wrong database {self.database}.Expected databases are sqlite ,mysql"
            )

            raise ValueError(
                f"Given wrong database {self.database}.Expected databases are sqlite ,mysql"
            )

    # def load_to_db(self, df: pd.DataFrame, table_name: str):
    #     """Loading the table data into databases."""
    #     # self.conn = self._connection_db()
    #     if self.database == "sqlite":
    #         self._sqlite_table_load(df, table_name)
    #     elif self.database == "mysql":
    #         self._mysql_table_load(df, table_name)

    def connection_db(self):
        """Connecting to database.

        Returns
        -------
        connection
            Connection object to the database.

        Raises
        ------
        ValueError
            If the database path doesn't exist.
        """

        if self.database == "sqlite":
            # # Check if the database file exists
            # if not os.path.isfile(self.config.db_params.sqlite_database_path):
            #     # If it doesn't exist, create an empty database file
            #     open(self.config.db_params.sqlite_database_path, 'w').close()

            if not self._fs.exists(self.data_config.db_params.sqlite_database_path):
                self.logger.error(
                    f"database path doesn't exist: {self.data_config.db_params.sqlite_database_path}"
                )
                raise ValueError("database path doesn't exist")

            self.logger.info("Connecting to SQLite database...")

            self.conn = load_sqlite3_database(
                path=self.data_config.db_params.sqlite_database_path,  fs=self._fs
            )

        elif self.database == "mysql":
            self.logger.info("Connecting to MySQL database...")
            self.conn = self._mysql_connection_db()

        return self.conn

    def _mysql_connection_db(self):
        try:
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(self.data_config.db_params.password_path),
            )
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                self.logger.error("Something is wrong with your user name or password")
            else:
                self.logger.error(f"Error connecting to MySQL: {err}")

        mycursor = conn.cursor()

        mycursor.execute("show databases")

        all_tables = mycursor.fetchall()

        if self.database_name is None:
            self.logger.error("Database name isn't given. Expected a string")

            raise ValueError("Database name isn't given. Expected a string")

        if (self.database_name,) in all_tables:
            self.logger.info(f"Connected to MySQL database: {self.database_name}")
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(self.data_config.db_params.password_path),
                database=self.database_name,
            )
        else:
            mycursor.execute(f"CREATE DATABASE {self.database_name}")
            self.logger.info(f"Created new database: {self.database_name}")
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(self.data_config.db_params.password_path),
                database=self.database_name,
            )
        return conn
