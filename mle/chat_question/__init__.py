import json
import logging

import azure.functions as func

from app.dtos.base import CustomJSONEncoder
from app.services import chat_service


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    try:
        print(req.get_body())
        req_body = req.get_json()

        user_id = req_body.get("user_id")
        chat_id = req_body.get("chat_id")
        context_id = req_body.get("context_id")
        question = req_body.get("question")

        result, err = chat_service.create_new_question(chat_id, context_id, user_id, question)
        if result is None:
            return func.HttpResponse(f"An error occurred while processing the request. {str(err)}", status_code=500)
        # Convert data to JSON string using the custom encoder
        json_data = json.dumps(result, cls=CustomJSONEncoder)

        return func.HttpResponse(json_data, status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"An error occurred while processing the request. {str(e)}", status_code=500)
