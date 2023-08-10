import json
import logging

import azure.functions as func

from app.dtos.base import CustomJSONEncoder
from app.dtos.chats import QuesByContextRequest
from app.services import chat_service


def main(req: func.HttpRequest) -> func.HttpResponse:  # get questions by context_id
    logging.info("Python HTTP trigger function processed a request.")

    req_body = req.get_json()
    context_id = req_body.get("context_id")
    logging.info(req_body)

    try:
        result = chat_service.get_questions_by_context_id(context_id)
        if result is None:
            return func.HttpResponse(f"An error occurred while processing the request.", status_code=500)

        # Convert data to JSON string using the custom encoder
        json_data = json.dumps(result, cls=CustomJSONEncoder)

        logging.info(json_data)

        return func.HttpResponse(json_data, status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"An error occurred while processing the request. {str(e)}", status_code=500)
