import pandas as pd


class SqlEvaluator:
    """
    Class for evaluating SQL queries.

    Attributes
    ----------
    conn : object
        Connection object for the database.

    Methods
    -------
    execute_query(query)
        Executes the provided SQL query and returns the result as a pandas dataframe.
    compare_tables(table1, table2)
        Compares two tables (in the form of dataframes) and returns True if they have the same columns and values,
        False otherwise.
    evaluate_queries(query1, df)
        Executes the provided SQL query and compares the result with the given dataframe.
        Returns True if they match, False otherwise.
    """

    def __init__(self, conn):
        """
        Initializes a SqlEvaluator object.

        Parameters
        ----------
        conn : object
            Connection object for the database.

        Returns
        -------
        None.

        """
        self.conn = conn

    def execute_query(self, query):
        """
        Executes the provided SQL query and returns the result as a pandas dataframe.

        Parameters
        ----------
        query : str
            SQL query to be executed.

        Returns
        -------
        pandas.DataFrame
            Result of the executed SQL query.

        """
        try:
            # TODO: Implement fsspec - This is not used anywhere. We can take this up later
            result = pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Error executing query: {query}\n{e}")
            result = pd.DataFrame()
        return result

    def compare_tables(self, table1, table2):
        """
        Compares two tables (in the form of dataframes) and returns True if they have the same columns and values,
        False otherwise.

        Parameters
        ----------
        table1 : pandas.DataFrame
            The first dataframe to be compared.
        table2 : pandas.DataFrame
            The second dataframe to be compared.

        Returns
        -------
        bool
            True if the dataframes have the same columns and values, False otherwise.

        """
        table1 = table1.sort_index(axis=1)
        table2 = table2.sort_index(axis=1)
        table1 = table1.sort_values(by=table1.columns.tolist()).reset_index(drop=True)
        table2 = table2.sort_values(by=table2.columns.tolist()).reset_index(drop=True)
        table1 = table1.round(2)
        table2 = table2.round(2)
        table1 = table1.applymap(lambda x: str(x))
        table2 = table2.applymap(lambda x: str(x))
        table1 = table1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        table2 = table2.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        if not table1.columns.equals(table2.columns):
            return False

        if not table1.equals(table2):
            return False

        return True

    def evaluate_queries(self, query1, df):
        """
        Executes the provided SQL query and compares the result with the given dataframe.
        Returns True if they match, False otherwise.

        Parameters
        ----------
        query1 : str
            SQL query to be executed.
        df : pandas.DataFrame
            The dataframe to be compared with the result of the SQL query.

        Returns
        -------
        bool
            True if the result of the SQL query matches the given dataframe, False otherwise.

        """
        df1 = self.execute_query(query1)
        return self.compare_tables(df1, df)
