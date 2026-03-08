# A minimal proxy server that just forwards the '/invocations' request to TGI, and handles /ping
import flask
import requests
from flask import Flask, Response

app = Flask(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    resp = requests.post(
        "http://localhost:8081/invocations",
        data=flask.request.data,
        headers=flask.request.headers,
        stream=True,
    )

    def generate():
        for line in resp.iter_lines():
            yield line + b"\n"

    return Response(generate(), status=resp.status_code, headers=dict(resp.headers))


@app.route("/ping", methods=["GET"])
def ping():
    try:
        resp = requests.get("http://localhost:8081/ping")
        return flask.Response(response=resp.text, status=resp.status_code, mimetype="application/json")
    except Exception:
        return flask.Response(response="\n", status=500, mimetype="application/json")
