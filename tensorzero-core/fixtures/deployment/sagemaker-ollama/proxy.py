# A minimal proxy server that just forwards the '/invocations' request to ollama, and handles /ping
import flask
import requests
from flask import Flask, Response

app = Flask(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    resp = requests.post(
        "http://localhost:11434/v1/chat/completions",
        data=flask.request.data,
        stream=True,
    )

    def generate():
        for line in resp.iter_lines():
            yield line + b"\n"

    return Response(generate(), status=resp.status_code, headers=dict(resp.headers))


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")
