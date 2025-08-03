FROM python:3.12-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN curl -fsSL https://ollama.com/install.sh -o ollama.sh
RUN bash ollama.sh

ADD . /app
WORKDIR /app

RUN uv sync --frozen --no-dev
ENTRYPOINT [ "./entrypoint.sh" ]

EXPOSE 80
