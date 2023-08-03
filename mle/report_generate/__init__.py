import json
import logging

import azure.functions as func

from app.dtos.base import CustomJSONEncoder
from app.services import chat_service, report_service


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()

        user_id = req_body.get("user_id")
        chat_id = req_body.get("chat_id")

        logging.info(req_body)

        report_details = report_service.generate_report(user_id, chat_id)
        if report_details is None:
            return func.HttpResponse(f"An error occurred while processing the request.", status_code=500)
        new_chat_details = chat_service.reset_chat(user_id)
        if new_chat_details is None:
            return func.HttpResponse(f"An error occurred while processing the request.", status_code=500)

        result = {"report": report_details, "new_chat": new_chat_details}

        # Convert data to JSON string using the custom encoder
        json_data = json.dumps(result, cls=CustomJSONEncoder)
        logging.info(json_data)

        return func.HttpResponse(json_data, status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"An error occurred while processing the request. {str(e)}", status_code=500)
