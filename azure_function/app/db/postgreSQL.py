import psycopg2
import json


def create_db_connection():
    try:
        # SQL authentication
        print("Establishing PostgreSQL DB connection ...")
        connection = psycopg2.connect(
            database="nlp_dev",
            user="supplyChainDbAdmin@supply-chain-db",
            password="scaitestuser12345#%",
            host="supply-chain-db.postgres.database.azure.com",
            port="5432",
        )
        return connection

    except Exception as err:
        print(f"Error: '{err}'")
        raise Exception("Database Connection Failed.")
        return null


def execute_query(connection, query, return_value: bool = False):
    """Execute DML queries"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        if return_value:
            primary_key_value = cursor.fetchone()[0]
            connection.commit()
            return primary_key_value
        else:
            connection.commit()
    except Exception as err:
        print("Query: ", query)
        print(f"Error: '{err}'")
        raise Exception("Database Query Execution Failed.")

def execute_query1(connection, query, content, return_value: bool = False):
    """Execute DML queries"""
    cursor = connection.cursor()
    try:
        ## Change starts
        c = json.dumps(content['Response JSON']).replace('\n',' ').replace("\'",' ')
        print("c"*30)
        print(c)
        cursor.execute(query, (content['Response JSON']['type'], c))
        ## Change ends
        if return_value:
            primary_key_value = cursor.fetchone()[0]
            connection.commit()
            return primary_key_value
        else:
            connection.commit()
    except Exception as err:
        print("Query: ", query)
        print(f"Error: '{err}'")
        raise Exception("Database Query Execution Failed.")


def read_query(connection, query: str, fetch_columns: bool = False, table_name: str = ""):
    """Read table data (DQL) using query"""
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        if fetch_columns:
            columns = list([row.column_name for row in cursor.columns(table=table_name)])
            return result, columns
        else:
            return result
    except Exception as err:
        print("Query: ", query)
        print(f"Error: '{err}'")
        raise Exception("Database Query Retrieval Failed.")
