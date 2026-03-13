import json

import wikipedia
from markdownify import markdownify

# ## Tools
#
# We define as Python functions the tools that will be used by the model.
#
# Here, we have a tool for searching Wikipedia and a tool for loading a Wikipedia page.
# These are also described in the config/tensorzero.toml file so that our agent automatically knows about them.


def search_wikipedia(arguments: dict) -> str:
    """
    Searches Wikipedia for a given query and returns a list of search results.

    Args:
        arguments: A dictionary containing the search query.
            Expected keys: {"query": str}

    Returns:
        A newline-separated list of Wikipedia search results.
    """
    query = arguments.get("query")
    if query is None:
        raise ValueError("The `query` argument is required for `search_wikipedia` tool.")

    return "\n".join(wikipedia.search(query))


def load_wikipedia_page(arguments: dict) -> str:
    """
    Loads and formats the content of a Wikipedia page.

    Args:
        arguments: A dictionary containing the page title.
            Expected keys: {"title": str}

    Returns:
        The formatted Wikipedia page content, or an error message if the page is not found.
    """
    title = arguments.get("title")
    if title is None:
        raise ValueError("The `title` argument is required for `load_wikipedia_page` tool.")

    try:
        page = wikipedia.page(title)
        # Preprocess result by converting the HTML content to Markdown to reduce token usage
        page_markdown = markdownify(page.html())
        return f"# URL\n\n{page.url}\n\n# CONTENT\n\n{page_markdown}"
    except wikipedia.exceptions.PageError:
        return f"ERROR: page '{title}' not found."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"ERROR: disambiguation error for '{title}': {e}"


def parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool call arguments from a JSON string."""
    try:
        return json.loads(arguments_str)
    except (json.JSONDecodeError, TypeError):
        return {}
