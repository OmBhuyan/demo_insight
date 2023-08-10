import logging

import mysql.connector
import pandas as pd
from mysql.connector import errorcode

from .utils import (
    fs_connection,
    get_fs_and_abs_path,
    load_db_credentials,
    load_sqlite3_database,
)

MYLOGGERNAME = "QueryInsights"


class DatabaseLoading:
    """Connecting to database and creating a table in that database

    Parameters
    ----------
    data_config : dict
        input data_config dictionary contains the paths to the data.
    database_name : str
        Database name where the table will be stored
    fs_key : str
        Account key for connecting to the File storage. If left as blank and platform specified in the data_config (data_config.cloud_storage.platform) is not blank, it will look for the path in the data_config and read the key from there, by default None
    """

    def __init__(
        self,
        data_config: dict,
        fs_key: str,
        database_name: str = None,
    ):
        self.logger = logging.getLogger(MYLOGGERNAME)

        self.data_config = data_config
        self.database = self.data_config.db_params.db_name
        self.database_name = database_name

        prefix_url, storage_options = fs_connection(
            fs_connection_dict=self.data_config.cloud_storage, fs_key=fs_key
        )
        self._fs, self.paths = get_fs_and_abs_path(
            path=prefix_url, storage_options=storage_options
        )

        list_of_databases = ["sqlite", "mysql"]

        if self.database not in list_of_databases:
            self.logger.error(
                f"Given wrong database {self.database}.Expected databases are sqlite ,mysql"
            )

            raise ValueError(
                f"Given wrong database {self.database}.Expected databases are sqlite ,mysql"
            )

    def connection_db(self):
        """Creates a connection to a database specified.

        Returns
        -------
        conn : sqlite3.Connection
            Connection object to the database.
        """
        if self.database == "sqlite":
            # # Check if the database file exists
            # if not self._fs.isfile(self.config.db_params.sqlite_database_path):
            #     # If it doesn't exist, create an empty database file
            #     open(self.config.db_params.sqlite_database_path, 'w').close()

            # Check if the database file exists
            if not self._fs.exists(self.data_config.db_params.sqlite_database_path):
                # If it doesn't exist, create an empty database file
                with self._fs.open(self.data_config.db_params.sqlite_database_path, "w") as _:
                    pass

            self.conn = load_sqlite3_database(
                path=self.data_config.db_params.sqlite_database_path, fs=self._fs
            )

        elif self.database == "mysql":
            self.conn = self._mysql_connection_db()

        return self.conn

    def load_to_db(self, df: pd.DataFrame, table_name: str):
        """Loading the table data into databases.

        Parameters
        ----------
        df : pd.DataFrame
            A dataset in the form of a dataframe that needs to be imported into database.
        table_name : str
            Table name for the imported dataframe in the database.

        Returns
        -------
        None
        """
        # self.conn = self._connection_db()
        if self.database == "sqlite":
            self._sqlite_table_load(df, table_name)
        elif self.database == "mysql":
            self._mysql_table_load(df, table_name)
        return None

    def _mysql_connection_db(self):
        """Creates a connection to a mysql database specified.

        Returns
        -------
        conn : mysql.connector.connection.MySQLConnection


        Raises
        ------
        ValueError
            If the database name is not given.

        """
        try:
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(
                    self.data_config.db_params.password_path, fs=self._fs
                ),
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
            self.logger.error("No Database name given.Expected a string")

            raise ValueError("No Database name given.Expected a string")

        if (self.database_name,) in all_tables:
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(
                    self.data_config.db_params.password_path, fs=self._fs
                ),
                database=self.database_name,
            )
        else:
            mycursor.execute(f"CREATE DATABASE {self.database_name}")
            conn = mysql.connector.connect(
                host=self.data_config.db_params.host,
                user=self.data_config.db_params.username,
                password=load_db_credentials(
                    self.data_config.db_params.password_path, fs=self._fs
                ),
                database=self.database_name,
            )
        return conn

    def _mysql_table_load(self, df, table_name):
        """Creates a table in the database and loads the dataframe into it.

        Parameters
        ----------
        df : pd.DataFrame
            A dataset in the form of a dataframe that needs to be imported into database.
        table_name : str
            Table name for the imported dataframe in the database.

        Returns
        -------
        None

        """
        df.fillna("None", inplace=True)
        column_names = df.columns.tolist()

        sql_data_types = {
            "object": "TEXT",
            "int64": "BIGINT",
            "float64": "REAL",
            "datetime64[ns]": "TIMESTAMP",
            "bool": "TINYINT",
        }

        columns = ", ".join(
            [
                f"{col_name} {sql_data_types[str(df[col_name].dtype)]} DEFAULT NULL"
                for col_name in column_names
            ]
        )

        try:
            if self.conn.is_connected():
                cursor = self.conn.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                self.logger.info(f"You're connected to database: {record}")
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                self.logger.info("Creating table....")
                cursor.execute(f"CREATE TABLE {table_name} ({columns});")
                self.logger.info(f"{table_name} table is created....")

                # Insert the DataFrame into a table
                df.to_sql(name=table_name, con=self.conn, if_exists="replace", index=False)

                # insert_values = ", ".join(
                #     [
                #         f"({', '.join([f'{repr(val)}' for val in row.values])})"
                #         for _, row in df.iterrows()
                #     ]
                # )

                # insert_query = (
                #     f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES {insert_values};"
                # )

                # cursor.execute(insert_query)
                # the connection is not autocommitted by default, so we must commit to save our changes
                self.conn.commit()
                cursor.close()
                self.conn.close()
                self.logger.info("Done.")

        except mysql.connector.Error as error:
            if error.errno == errorcode.OPERATIONAL_ERROR:
                self.logger.error("Operational error: {}".format(error))
            elif error.errno == errorcode.PROGRAMMING_ERROR:
                self.logger.error("Programming error: {}".format(error))
            elif error.errno == errorcode.INTEGRITY_ERROR:
                self.logger.error("Integrity error: {}".format(error))
            elif error.errno == errorcode.DATA_ERROR:
                self.logger.error("Data error: {}".format(error))
            elif error.errno == errorcode.NOT_SUPPORTED_ERROR:
                self.logger.error("Not supported error: {}".format(error))
            else:
                self.logger.error("Error: {}".format(error))

        return None

    def _sqlite_table_load(self, df, table_name):
        """Creates a table in the database and loads the dataframe into it.

        Parameters
        ----------
        df : pd.DataFrame
            A dataset in the form of a dataframe that needs to be imported into database.
        table_name : str
            Table name for the imported dataframe in the database.

        Returns
        -------
            None

        """
        cursor = self.conn.cursor()

        chunk_size = self.data_config.db_params.chunk_size

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

        cursor.execute("PRAGMA journal_mode=OFF")

        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        self.logger.info("Creating table....")
        create_table_query = f"CREATE TABLE {table_name} (\n{columns}\n);"
        self.logger.info(f"{table_name} table is created....")

        cursor.execute(create_table_query)

        # Calculate the number of chunks based on the DataFrame size and chunk_size
        num_chunks = int(df.shape[0] / chunk_size) + 1

        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, df.shape[0])

            # Get the chunk of data from the DataFrame
            chunk = df.iloc[start_index:end_index]

            # Insert the chunk into the table
            chunk.to_sql(name=table_name, con=self.conn, if_exists="append", index=False)

        # Insert the DataFrame into a table
        # df.to_sql(name=table_name, con=self.conn, if_exists="replace", index=False)

        # insert_values = ", ".join(
        #     [f"({', '.join([f'{repr(val)}' for val in row.values])})" for _, row in df.iterrows()]
        # )
        # insert_query = (
        #     f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES {insert_values};"
        # )

        # cursor.execute(insert_query)

        # cursor.execute("""SELECT name FROM sqlite_master WHERE type='table';""")

        # self.logger.debug(f"{cursor.fetchall()}")

        # the connection is not autocommitted by default, so we must commit to save our changes
        self.conn.commit()
        # cursor.close()
        # self.conn.close()
        # self.logger.info("Done.")

        return None
