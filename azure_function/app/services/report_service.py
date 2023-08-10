import traceback
from datetime import datetime
from typing import List
import logging
logging.getLogger().setLevel(logging.INFO)

from app.db.postgreSQL import create_db_connection, execute_query, read_query
from app.dtos.report import GenerateReportResponse
from app.services.chat_service import get_chat_history_by_chat_id


def create_new_report(chat_id: str, user_id: str, db_conn=None):
    try:
        if db_conn == None:
            db_conn = create_db_connection()

        query = f"SELECT COUNT(DISTINCT report_id) AS counts FROM chat_reports;"
        count = read_query(db_conn, query)
        report_name = f"Report_{count[0][0]}"
        report_url = f"s3://path/to/{report_name}"

        query = f"""INSERT INTO chat_reports (report_name, chat_id, ques_list, created_time, user_id, report_url)
            VALUES
            ('{report_name}', {chat_id}, NULL, CURRENT_TIMESTAMP, '{user_id}', '{report_url}') RETURNING report_id;"""
        report_id = execute_query(db_conn, query, True)
        created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TODO: get created time from above excute_query itself.
        return report_id, report_name, report_url, created_time
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return None


def generate_report(user_id: str, chat_id: str, db_conn=None):
    try:
        report_details = GenerateReportResponse(chat_id=chat_id, user_id=user_id)
        if db_conn == None:
            db_conn = create_db_connection()

        # 1. Generate report
        # Create new report entry in chat_report table
        (
            report_details.report_id,
            report_details.report_name,
            report_details.report_url,
            report_details.created_time,
        ) = create_new_report(chat_id, user_id, db_conn)

        # Get reports chat_history
        report_details.chat_history = get_chat_history_by_chat_id([chat_id], db_conn)

        return report_details
        # TODO: Create PDF for Report
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return None
