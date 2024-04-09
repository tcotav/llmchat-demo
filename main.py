from flask import Flask, request, jsonify, render_template # type:ignore
from werkzeug.utils import secure_filename 
from flask_sslify import SSLify
import os
import bpschat

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
sslify = SSLify(app)


def log_error(message, extra):
    app.logger.error(message, extra)

def log_info(message, extra):
    app.logger.info(message, extra)

def return_json_error(message, code, detail=""):
    logging_dict={"error": message, "detail": detail, "code": code}
    log_error(message, extra=logging_dict)
    return jsonify({"error": message, "detail": detail}), code

@app.route('/')
def index():
    return render_template('index.html')

"""
expected format of the chat is:
{'text': 'Hello, World!'}
"""
@app.route('/chat', methods=['POST'])
def chat_msg():
    # validate the json data

    try:
        json_data=request.get_json()
        log_info("chat_msg", {"json_data": json_data})
        if json_data is None:
            return return_json_error("invalid input", 400, "error: no json data")
        # now we want to work with the question
        # {'conversation': [{'role': 'user', 'content': "What is Blue's favorite drink?"}]}
        conversation_history = json_data['conversation']
        question=conversation_history[-1]['content']

        response=bpschat.ask_document_with_state(1, question)
        print("in /chat, response is: ", response)

        log_info("chat_msg", {"response": response})
        return jsonify({"message": response}) 
    except Exception as e:
        print(e)
        return return_json_error("invalid input", 400, "error:{}".format(str(e)))
    



if __name__ == '__main__':
    # check if open ai environment parameter is set using os
    if os.getenv('OPENAI_API_KEY') is None:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    app.run(host="0.0.0.0", port='5002', debug=True, ssl_context='adhoc')