import json
import logging

import azure.functions as func

from app.dtos.base import CustomJSONEncoder
from app.services import chat_service


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()

        answer_id = req_body.get("answer_id")
        status = req_body.get("status")
        reason = req_body.get("reason")

        logging.info(req_body)

        result = chat_service.feedback(answer_id, status, reason)
        if result is None:
            return func.HttpResponse(f"An error occurred while processing the request.", status_code=500)


        # Convert data to JSON string using the custom encoder
        json_data = json.dumps(result, cls=CustomJSONEncoder)

        logging.info(json_data)

        return func.HttpResponse(json_data, status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"An error occurred while processing the request. {str(e)}", status_code=500)
