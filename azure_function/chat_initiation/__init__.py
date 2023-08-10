import json
import logging

# import os
import azure.functions as func

from app.dtos.base import CustomJSONEncoder
from app.dtos.chats import CreateOrUpdateChatRequest
from app.services import chat_service


# azure function for initiating a chat
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    try:
        print(req)
        req_body = req.get_json()
        user_id = req_body.get("user_id")
        logging.info(req_body)
        # logging.info("TESTTTTTTTTTTTTTTT", os.getenv("test"))
        if user_id:
            request = CreateOrUpdateChatRequest(user_id=user_id)
            result = chat_service.check_existing_user(request.user_id)
            if result is None:
                return func.HttpResponse(f"An error occurred while processing the request.", status_code=500)

            # Convert data to JSON string using the custom encoder
            json_data = json.dumps(result, cls=CustomJSONEncoder)

            logging.info(json_data)

            return func.HttpResponse(json_data, status_code=200, mimetype="application/json")
        else:
            return func.HttpResponse("User ID is missing in the request body.", status_code=400)

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(
            f"An error occurred while processing the request. {str(e)}", status_code=500
        )
