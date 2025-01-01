# TensorZero Jeopardy Example

## Overview
This project demonstrates how to:
1. Query Wikipedia for context using the `retrieve_context` function.
2. Answer Jeopardy-style questions using the `answer_question` function.

## Setup

### Prerequisites
- Docker and Docker Compose installed.
- Python 3.x installed with the following libraries:
  ```bash
  pip install wikipedia requests


tensorzero-jeopardy-example/
├── config/
│   ├── tensorzero.toml
│   ├── functions/
│       ├── retrieve_context/
│       │   ├── user_schema.json
│       │   ├── system_schema.json
│       │   ├── system_template.minijinja
│       │   ├── user_template.minijinja
│       ├── answer_question/
│           ├── user_schema.json
│           ├── system_schema.json
│           ├── system_template.minijinja
│           ├── user_template.minijinja
├── docker-compose.yml
├── .env
├── scripts/
│   ├── wikipedia_query.py
│   ├── evaluate_models.py
├── jeopardy_dataset.json
└── README.md


docker-compose up --build


curl -X POST http://localhost:3000/v1/functions/retrieve_context \
     -H "Content-Type: application/json" \
     -d '{"topic": "TensorFlow"}'


curl -X POST http://localhost:3000/v1/functions/answer_question \
     -H "Content-Type: application/json" \
     -d '{"question": "What is TensorFlow?", "context": "TensorFlow is a machine learning framework developed by Google."}'


python scripts/evaluate_models.py
