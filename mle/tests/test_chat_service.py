import sys
from os.path import abspath, dirname

# Add the parent directory of the test module to the Python path
sys.path.append(dirname(dirname(abspath(__file__))))

# Import the necessary modules from the app package
from app.services.chat_service import *

def test_create_new_chat():
    db_conn = None  # Initialize db_conn variable if needed
    user_id = "sai"
    new_chat_id, new_chat_name = create_new_chat(user_id, db_conn)
    chat_id = 17
    chat_name = "chat YB6QA5"
    assert new_chat_id == chat_id
    assert new_chat_name == chat_name


# def test_get_report_list(user_id: str, db_conn=None):
#     if db_conn == None:
#             db_conn = create_db_connection()
#     user_id = "sai"
#     new_chat_id, new_chat_name = create_new_chat(user_id, db_conn)
#     chat_id = 17
#     chat_name= "chat YB6QA5"
#     assert new_chat_id == chat_id
#     assert new_chat_name == chat_name
