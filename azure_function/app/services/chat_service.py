import random
import string
import traceback
from datetime import datetime
from typing import List, Optional
import json
import logging
logging.getLogger().setLevel(logging.INFO)
import pickle
import jwt
import pdb
import math
import os

from query_insights.api import QueryInsights

# from app.query_insights.api import QueryInsights
from app.db.postgreSQL import create_db_connection, execute_query, read_query, execute_query1
from app.dtos.chats import CreateOrUpdateChatResponse, TopicResponse
from app.dtos.question import QuestionResponse
from app.dtos.report import GenerateReportResponse

# TODO: Implement decorate for @decoratorInitDBIfNotProvided. means you don't have write redundant code in each function to initiate database connetion if not provided
# TODO: Add docstring for all functions
# TODO: Comment all code properly
# TODO: make easy to understand name for functions, variable and dtos entity

access_token = "eyJ0eXAiOiJKV1QiLCJub25jZSI6ImEtdHhtLWoxMUxuZTd0bmlFbHdWNXhsNDF5OEdROEpCN3ZxdlRYUFZZOGciLCJhbGciOiJSUzI1NiIsIng1dCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyIsImtpZCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyJ9.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTAwMDAtYzAwMC0wMDAwMDAwMDAwMDAiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC9lNzE0ZWYzMS1mYWFiLTQxZDItOWYxZS1lNmRmNGFmMTZhYjgvIiwiaWF0IjoxNjg2NzEwMjYyLCJuYmYiOjE2ODY3MTAyNjIsImV4cCI6MTY4NjcxNTM2OSwiYWNjdCI6MCwiYWNyIjoiMSIsImFpbyI6IkFUUUF5LzhUQUFBQW9jWkZTWmlCVjYrYUE2Y29NWXdNbkljZkdhREc3YThZVnVINVAxVTR4alF4dEdrNjVOS0xIVzFBSU5LTGZjdTIiLCJhbXIiOlsicHdkIl0sImFwcF9kaXNwbGF5bmFtZSI6InN1cHBseWNoYWluLXNwbiIsImFwcGlkIjoiZTM0NjAwNmYtMWJhZC00ZmFlLTkwNjEtM2QwYTEzZmExMTdiIiwiYXBwaWRhY3IiOiIwIiwiaWR0eXAiOiJ1c2VyIiwiaXBhZGRyIjoiMjEwLjE4LjE3OS4yMTgiLCJuYW1lIjoiS2lzaG9yZXNhaSBHYW5lc2hrdW1hciIsIm9pZCI6IjQyNTJlMjFmLWQzM2YtNGU1NS04NjRhLTEwMzA2NWI0YmMwZiIsInBsYXRmIjoiMyIsInB1aWQiOiIxMDAzMjAwMUREQ0Y3NTg5IiwicmgiOiIwLkFWWUFNZThVNTZ2NjBrR2ZIdWJmU3ZGcXVBTUFBQUFBQUFBQXdBQUFBQUFBQUFCV0FPVS4iLCJzY3AiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsInNpZ25pbl9zdGF0ZSI6WyJrbXNpIl0sInN1YiI6Imp6VEtTLWxRUXRvVzlqcWk4Qy0yQklXc1YtMkdLSWFrTWZ5UU1sb1FOVEkiLCJ0ZW5hbnRfcmVnaW9uX3Njb3BlIjoiQVMiLCJ0aWQiOiJlNzE0ZWYzMS1mYWFiLTQxZDItOWYxZS1lNmRmNGFmMTZhYjgiLCJ1bmlxdWVfbmFtZSI6Imtpc2hvcmVzYWkuZ2FuZXNAdGlnZXJhbmFseXRpY3MuY29tIiwidXBuIjoia2lzaG9yZXNhaS5nYW5lc0B0aWdlcmFuYWx5dGljcy5jb20iLCJ1dGkiOiJPejhTUzMyekEwdVlBcDAyNk5jRkFBIiwidmVyIjoiMS4wIiwid2lkcyI6WyJiNzlmYmY0ZC0zZWY5LTQ2ODktODE0My03NmIxOTRlODU1MDkiXSwieG1zX3N0Ijp7InN1YiI6Ilo3ZlRvUHc3M3VOOEJiOE9FelhVcmU0R2ZzVHBQY2NhcG1LaUZEWEpHZ1kifSwieG1zX3RjZHQiOjE0NzE0MTUyMzl9.jjGQlgKtwEM9NfuIMKzbCcmsoDrj7LA6SfvmTTG0dLQd2l9TpxtuqHhBr8JDcQAccUrV4UyrzX6gIaU0Yy7uvzA-XymkV1v-A42JV7e_T10e2NG1fa_ESmAAADtVX1xjmh8yznvc0oExy_eHZgEGwvyzORfDGuQZezLupQbzx7BhwKpvhOw3vqIPkiYMNtqmJzraPzqRUCn1BgJC-rYquim-tR0gQDnT7cSdnj6RdsQlo7dxXBYzDvwbF3tr8O10tBaaeyClkbv6GCssVuY9WdUh_zP8r2nWWXS6v3OJt-lVkkBWBVyhF6o9Oql0xu_n-YqCJl2PpLCKzGhuqQAMig"
data_config_path = "./configs/local/data_config_azure.yaml"
user_config_path = "./configs/local/user_config.yaml"
model_config_path = "./configs/model_config.yaml"
debug_config_path = "./configs/debug_code_config.yaml"

api_key = os.environ.get("azure_OPENAI_API_KEY")
fs_key = os.environ.get("azure_BLOB_ACCOUNT_KEY")

api_key_c = os.environ.get("CUSTOMCONNSTR_azure_OPENAI_API_KEY")
fs_key_c = os.environ.get("CUSTOMCONNSTR_azure_BLOB_ACCOUNT_KEY")


if api_key is None:
    api_key = api_key_c
if fs_key is None:
    fs_key = fs_key_c

if fs_key is None:
    exit()

qi = QueryInsights(
    user_config_path=user_config_path,
    data_config_path=data_config_path,
    model_config_path=model_config_path,
    debug_config_path=debug_config_path,
    api_key=api_key,
    fs_key=fs_key,
    logging_level="INFO",
)


def create_new_chat(user_id: str, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - user_id: Specifies the user ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns Chat ID and Chat Name
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        chat_name = "chat " + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

        query = f"""UPDATE chat_details SET is_active=FALSE WHERE user_id='{user_id}';
                    INSERT INTO chat_details
                    (user_id, chat_name, is_active, created_time, modified_time)
                    VALUES
                    ('{user_id}', '{chat_name}', TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING chat_id;"""

        chat_id = execute_query(db_conn, query, True)
        return chat_id, chat_name
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return None


def get_report_list(user_id: str, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - user_id: Specifies the user ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns a list containing result objects with chat and report information related details.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        query = f"""select * from chat_reports where user_id='{user_id}';"""
        report_rows = read_query(db_conn, query)
        result = []
        for row in report_rows:
            item = GenerateReportResponse()
            item.report_id = row[0]
            item.report_name = row[1]
            item.chat_id = row[2]
            item.ques_list = row[3] if row[3] != None else []
            item.created_time = row[4]
            item.user_id = row[5]
            item.report_url = row[6]
            result.append(item)

        return result
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return None


def reset_chat(user_id: str, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - user_id: Specifies the user ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the result object containing new chat details and updated report history.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        result = CreateOrUpdateChatResponse(user_id=user_id, chat_details={})
        # 1. Create New chat
        new_chat_id, new_chat_name = create_new_chat(user_id, db_conn)

        # 2. Create New Context
        new_context_id = (
            create_new_context(new_chat_id, user_id, db_conn) if new_chat_id != None else None
        )

        # 3. Get Report History
        result.report_list = get_report_list(user_id, db_conn)

        # 4. Get chat History
        result.chat_details.chat_history = {}
        result.chat_details.chat_id = new_chat_id
        result.chat_details.chat_name = new_chat_name
        result.chat_details.is_active = True
        result.chat_details.context_id = new_context_id

        return result

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def check_existing_user(user_id: str):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - user_id: Specifies the user ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the result object containing the updated chat details and report history for the user.
    """
    try:
        result = CreateOrUpdateChatResponse(user_id=user_id, chat_details={})
        # Check for existing user
        db_conn = create_db_connection()
        query = f"SELECT * FROM user_config WHERE user_id='{user_id}';"
        user_rows = read_query(db_conn, query)

        # TODO: also add last_login_time column which you will get from get_user_details_from_token
        # last_login_time = get_last_login_time(access_token)
        last_login_time = "CURRENT_TIMESTAMP"

        if len(user_rows) > 0:
            # Functionality for exisitng user
            # Update user's login time
            query = f"UPDATE user_config SET last_login_time = {last_login_time}, modified_time = CURRENT_TIMESTAMP WHERE user_id='{user_id}';"
            execute_query(db_conn, query)

            # get chat details
            query = f"""SELECT cd.chat_id, cd.chat_name, cd.is_active, cc.context_id
                    FROM chat_details AS cd
                    LEFT JOIN chat_contexts AS cc ON cc.chat_id = cd.chat_id
                    WHERE cd.user_id = '{user_id}' AND cd.is_active = TRUE AND cc.is_active=TRUE;"""
            chat_details_row = read_query(db_conn, query)[0]
            result.chat_details.chat_id = chat_details_row[0]
            result.chat_details.chat_name = chat_details_row[1]
            result.chat_details.is_active = chat_details_row[2]
            result.chat_details.context_id = chat_details_row[3]

            # Get report
            result.report_list = get_report_list(user_id, db_conn)
            # TODO: Implemente Get chat history functionality
            result.chat_details.chat_history = get_chat_history_by_chat_id(
                [result.chat_details.chat_id], db_conn
            )
        else:
            # Functionality for New user

            # TODO: Add user_name, User_email columns in user_config table in database. ```name, user_id, email = get_user_details_from_token(token)```
            query = f"INSERT INTO user_config (user_id, created_time, modified_time, last_login_time) VALUES ('{user_id}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, {last_login_time});"
            execute_query(db_conn, query)

            result = reset_chat(user_id, db_conn)

        return result

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def get_last_login_time(access_token):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - access_token: Specifies the access token of the user

    Returns
    -------
        Returns the last_login_time of the user.
    """
    try:
        # TODO: Implement GettingTimeFromJwtToken When we start getting Token in Request Header from frontend team
        # decoded_token = jwt.decode(access_token, verify=False, algorithms=['HS256'])  # Use appropriate decoding and verification
        iat_timestamp = datetime.now()  # decoded_token.get("iat")
        formatted_time = iat_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return {"error": True}


def get_scopes_from_token(access_token):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - access_token: Specifies the access token of the user

    Returns
    -------
        Returns the scopes of the user.
    """
    try:
        decoded_token = jwt.decode(
            access_token, verify=False
        )  # Use appropriate decoding and verification
        scopes = decoded_token.get("scp", "").split()
        return scopes
    except jwt.DecodeError:
        # Handle decoding error
        return []
    except jwt.ExpiredSignatureError:
        # Handle expired token error
        return []


def replace_nan_with_none(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) or isinstance(value, list):
                replace_nan_with_none(value)
            elif isinstance(value, float) and math.isnan(value):
                data[key] = None
    elif isinstance(data, list):
        for item in data:
            replace_nan_with_none(item)


def create_new_question(chat_id: int, context_id: int, user_id: str, question: str, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - chat_id: Specifies the chat ID
        - context_id: Specifies the context ID
        - user_id: Specifies the user ID
        - question: Specifies the question asked by the user
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the result object containing the updated chat details for the user.
    """
    try:
        result = QuestionResponse()
        if db_conn == None:
            db_conn = create_db_connection()
        # Steps
        # TODO: make get_answer_from_question function and move code there
        # "Ques1; Ques2; Ques3"
        ques_list = get_questions_by_context_id(context_id)
        if ques_list is None:
            ques_str = question
        else:
            # ques_list = list(set(ques_list))
            logging.info(len(ques_list))
            bots=get_bot_response(ques_list,context_id)
            
            ques_list=[list(pair) for pair in zip(ques_list, bots)]
            ques_list.append([question,None])
            # ques_str = "; ".join([f"{pair[0]}: {pair[1]}" for pair in ques_list])
        logging.info(ques_list)
        # logging.info(ques_str)
        # 1. call DS API to get answer for given question
        # qi = QueryInsights(
        # user_config_path=user_config_path,
        # data_config_path=data_config_path,
        # model_config_path=model_config_path,
        # debug_config_path=debug_config_path,
        # logging_level="INFO",
        # )
        queryy, chart, insights, content = qi.run_query_insights(
            question=ques_list, additional_context=None
        )
        logging.info(content)
        logging.info(content['Response JSON']['response_for_history'])
        # content = send_mock_insight_type_answer()
        # category = "insight"
        # TODO: get category from DS API only

        # 2. store question in database
        # Store answer in chat_response
        print("*" * 30)
        print(content)
        replace_nan_with_none(content)
        print("#" * 30)
        print(content["Response JSON"])
        print("@" * 30)
        query = """
            INSERT INTO chat_responses (category, created_time, modified_time, content)
            VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s)
            RETURNING answer_id;
        """
        answer_id = execute_query1(db_conn, query, content, True)
        print("answer_id", answer_id)
        question_to_insert = question.replace("'", "''")
        response_to_insert = content['Response JSON']['response_for_history'].replace("'", "''")
        # store question data and answer_id in chat_questions
        query = f"""INSERT INTO chat_questions
                (user_question, answer_id, context_id, chat_id, user_id, created_time, modified_time,response_for_history)
                VALUES
                ('{question_to_insert}', {answer_id}, {context_id}, {chat_id}, '{user_id}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,'{response_to_insert}') RETURNING question_id;"""
        question_id = execute_query(db_conn, query, True)
        print("question_id", question_id)
        # 3. modified DS API's response in such way it could be provided to frontend Team
        result.user_id=user_id
        result.question_id = question_id
        result.answer_id = answer_id
        result.category = content["Response JSON"]["type"]
        # TODO: Get created_time from above question-storing-query
        result.content = content["Response JSON"]
        print("Result mil gya", result)
        return result, None

    except Exception as e:
        logging.info(traceback.format_exc())
        logging.info(e)
        return None, e

def get_bot_response(ques_list:list, context_id:int, db_conn=None):
    try:
        if db_conn == None:
            db_conn = create_db_connection()
        tuple_ques_list = tuple(ques_list)
        logging.info(tuple_ques_list)
        query = f"""SELECT ARRAY_AGG(response_for_history) AS response_for_history_list
                    FROM chat_questions
                    WHERE user_question IN {tuple_ques_list} and context_id={context_id};"""

        bots = read_query(db_conn, query)
        logging.info(bots[0][0])
        return bots[0][0]
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None

def create_new_context(chat_id: str, user_id: str, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - chat_id: Specifies the chat ID
        - user_id: Specifies the user ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the context ID created for the user.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        query = f"""UPDATE chat_contexts SET is_active=FALSE WHERE user_id='{user_id}';
                    INSERT INTO chat_contexts (user_id, chat_id, created_time, modified_time)
                    VALUES ('{user_id}', {chat_id}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING context_id;"""

        context_id = execute_query(db_conn, query, True)
        return context_id
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def get_chat_history_by_chat_id(chat_ids: List[str], db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - chat_ids: Specifies the list of chat IDs
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the chat history of the user using the list of chat IDs.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()
        query = f"""
                    SELECT json_object_agg(chat_history.context_id, chat_history.context_data) as response
                        FROM (
                            SELECT chat_history.context_id, json_agg(chat_history) as context_data
                            FROM (
                                SELECT cq.question_id, cq.answer_id, cq.context_id, cr.category as type, cr.user_feedback as feedback, cq.created_time, cr.answer_seq, cq.user_question as question, cr.content as data
                                FROM chat_questions AS cq
                                LEFT JOIN chat_responses AS cr ON cq.answer_id = cr.answer_id
                                WHERE cq.chat_id IN ({','.join(map(str, chat_ids))})
                                ORDER BY cq.question_id
                            ) AS chat_history
                            GROUP BY chat_history.context_id
                            ORDER BY chat_history.context_id
                        ) AS chat_history
                    """
        chat_history = read_query(db_conn, query)

        return chat_history[0][0] if chat_history[0][0] else {}

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def feedback(answer_id: str, status: bool, reason: Optional[str] = None, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - answer_id: Specifies the answer ID
        - status: Specifies the status given by the user for an answer (if status=True means user liked the answer, if status=False means user disliked the answer).
        - reason: Specifies the reason if the user disliked the answer (status=False)
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the context ID created for the user.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        query = f"UPDATE chat_responses SET user_feedback={status}, user_comment='{reason}', modified_time = CURRENT_TIMESTAMP WHERE answer_id='{answer_id}';"
        execute_query(db_conn, query)
        return {"status": True}

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def get_questions_by_context_id(context_id: int, db_conn=None):
    """
    Creates a database connection, if it doesn't exist.

    Parameters
    ----------
        - context_id: Specifies the context ID
        - db_conn: Specifies the database connection

    Returns
    -------
        Returns the list of questions for the given context ID.
    """
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        query = f"SELECT array_agg(user_question) as question_list FROM chat_questions WHERE context_id = {context_id};"
        queslist = read_query(db_conn, query)
        return queslist[0][0]

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None


def send_mock_insight_type_answer():
    # TODO: get a second response without an error
    return """jsonb_build_array(
                jsonb_build_object(
                'insight_type', 'sql_query',
                'content', 'SELECT COUNT(DISTINCT sto_sap_invoice) AS shipment_count FROM invoice_data WHERE source_location_name LIKE ''%Location 14%''',
                'error', null,
                'showError', false
                ),
                jsonb_build_object(
                'insight_type', 'chart',
                'content', '{"x-axis": "shipment1, shipment2, shipment3", "y-axis": "34, 45, 67"}',
                'error', null,
                'showError', true
                ),
                jsonb_build_object(
                'insight_type', 'summary',
                'content', 'The table contains data for three brands: BoldBites, FlavorCrave, and RidgeBite, with ordered_qty values of 44,879,360, 37,633,471, and 24,690,799 respectively, and all having an unshipped_qty of 4. - There is a strong positive correlation between brand popularity and ordered_qty, with BoldBites being the most popular and RidgeBite being the least popular.',
                'error', null,
                'showError', false
                )
            )"""


def send_mock_clarification_type_answer():
    return ({"content": "I did not understand. Can you give me more business context?"},)


def send_mock_table_selector_type_answer():
    return ({"content": "[{table_json1}, {table_json2}, {table_json3}, {table_json4}]"},)
