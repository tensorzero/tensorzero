# A simple container with Python 3.10 and the TensorZero client library

FROM python:3.10-slim-bookworm

WORKDIR /app

COPY clients/python /app/tensorzero/python

RUN pip install -e /app/tensorzero/python

CMD ["python", "script.py"]
