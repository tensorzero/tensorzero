# Example: Simple Agentic RAG — Multi-Hop Question Answering with LLMs

## Background

We present a simple example of a multi-hop retrieval agent using TensorZero.
The agent iteratively searches Wikipedia to gather information, and decides when it has enough context to answer a complex question.

The agent has access to four tools:

- `search_wikipedia`: search Wikipedia for a given query.
- `load_wikipedia_page`: load the content of a Wikipedia page.
- `think`: think about the question and the information available.
- `answer_question`: answer the question.

We encourage the agent to use the `think` tool to plan its approach to the question.
See [this article](https://www.anthropic.com/engineering/claude-think-tool) from Anthropic showing that such a tool often improves the quality of agentic workflows.

Even with this simple setup, we see that the agent can solve complex queries like: _"What is a common dish in the hometown of the scientist that won the Nobel Prize for the discovery of the positron?"_

Here's a sample response trace:

```
[Tool Call] think: {'thought': 'To answer the question, I need to first identify the scientist who won the Nobel Prize for the discovery of the positron. After that, I will determine his hometown and look for a common dish associated with that place.'}

[Tool Call] search_wikipedia: {'query': 'Nobel Prize positron discovery'}

[Tool Call] load_wikipedia_page: {'title': 'Carl David Anderson'}

[Tool Call] think: {'thought': 'Carl David Anderson, the scientist who discovered the positron, was born in New York City. I need to find common dishes that are characteristic of New York City.'}

[Tool Call] search_wikipedia: {'query': 'Cuisine of New York City'}

[Tool Call] load_wikipedia_page: {'title': 'Cuisine of New York City'}

[Tool Call] think: {'thought': 'I have established that Carl David Anderson, who won the Nobel Prize for discovering the positron, was born in New York City. Based on the information from the cuisine of New York City, some iconic dishes include New York-style pizza, bagels with lox, and pastrami on rye. I need to decide on one common dish to summarize.'}

[Tool Call] answer_question: {'answer': 'The scientist who discovered the positron was Carl David Anderson, born in New York City. A common dish associated with New York City is the New York-style bagel, often served with cream cheese and lox. This iconic dish reflects the city\'s rich culinary diversity, particularly its Jewish heritage.\n\nFor more information, see the Wikipedia pages on [Carl David Anderson](https://en.wikipedia.org/wiki/Carl_David_Anderson) and [Cuisine of New York City](https://en.wikipedia.org/wiki/Cuisine_of_New_York_City).'}
```

## Getting Started

### Prerequisites

1. Install [Docker](https://www.docker.com/).
2. Install Python 3.10+.
3. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Create a `.env` file with your API key (see `.env.example`):

   ```bash
   cp .env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

2. Start the services (ClickHouse, Postgres, TensorZero Gateway, and UI):

   ```bash
   docker compose up
   ```

3. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

   ```bash
   uv sync
   ```

4. Run the `main.ipynb` Jupyter notebook.

### Exploring Results

Once the services are running, you can explore the inferences and analytics in the TensorZero UI at [http://localhost:4000](http://localhost:4000).
