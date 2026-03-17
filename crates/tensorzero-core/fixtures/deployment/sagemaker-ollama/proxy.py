# A minimal proxy server that just forwards the '/invocations' request to ollama, and handles /ping
import json

import flask
import requests
from flask import Flask, Response

app = Flask(__name__)

NUM_CTX = 8192


@app.route("/invocations", methods=["POST"])
def invocations():
    body = json.loads(flask.request.data)
    body.setdefault("num_ctx", NUM_CTX)
    resp = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json=body,
        stream=True,
    )

    def generate():
        for line in resp.iter_lines():
            yield line + b"\n"

    return Response(generate(), status=resp.status_code, headers=dict(resp.headers))


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")
