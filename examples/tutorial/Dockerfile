# A simple container with Python 3.10 and the TensorZero client library

FROM python:3.10-slim-bookworm

WORKDIR /app

COPY clients/python-asyncio /app/tensorzero/python-asyncio

RUN pip install -e /app/tensorzero/python-asyncio

CMD ["python", "script.py"]
