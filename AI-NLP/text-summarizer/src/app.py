import json
import logging
import sys

from anthem.tenx.utils.config import GLOBAL_CONFIG
from anthem.tenx.utils.log import SplunkLogger
from flask import Flask, jsonify, make_response, request

logging.setLoggerClass(SplunkLogger)

app = Flask(__name__)
logger = logging.getLogger("app")


@app.route("/hello")
def hello():
    app.logger.info("Hello World found")
    logger.info("message", thomas="cool")
    return "Hello World"


@app.route("/configprops")
def configprops():
    logger.info("SPLUNK-TRACE", request="GET", func="myfunc")
    print(GLOBAL_CONFIG)
    return GLOBAL_CONFIG


@app.route("/refresh", methods=["POST"])
def refresh():
    return GLOBAL_CONFIG.load()


@app.route("/ping", methods=["GET"])
def ping():
    """
    Check if API Alive
    ---
    tags:
      - Check Alive
    consumes:
      - application/json
    produces:
      - application/json
      - text/xml
      - text/html
    responses:
        200:
            description: Success
            schema:
            id: return_test
            properties:
        500:
            description: Error
    """

    return jsonify({"status": "Alive"}), 200


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
if __name__ == "__main__":
    app.run()
