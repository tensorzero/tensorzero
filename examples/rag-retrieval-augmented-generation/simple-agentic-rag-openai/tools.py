"""
Wikipedia tools for the simple agentic RAG example using OpenAI Agents SDK.
These tools are implemented as function_tools that can be used by agents.
"""

import wikipedia
from markdownify import markdownify
from agents import function_tool


@function_tool
def think(thought: str) -> str:
    """
    Think about the question and the information you have gathered so far.
    This is a good time to plan your next steps.

    Args:
        thought: Your thoughts on the question and the information you have gathered so far.

    Returns:
        Empty string (this tool is for reasoning, not output)
    """
    # The think tool is just for reasoning - we return empty string
    # The Agents SDK will handle this automatically
    return ""


@function_tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for pages that match the query. Returns a list of page titles.

    Args:
        query: The query to search Wikipedia for (e.g. "machine learning").

    Returns:
        Newline-separated list of Wikipedia search results.
    """
    try:
        search_results = wikipedia.search(query)
        return "\n".join(search_results)
    except Exception as e:
        return f"ERROR: Wikipedia search failed: {e}"


@function_tool
def load_wikipedia_page(title: str) -> str:
    """
    Load a Wikipedia page. Returns the page content, or an error if the page does not exist.

    Args:
        title: The title of the Wikipedia page to load.

    Returns:
        The Wikipedia page content in Markdown format, including URL.
    """
    try:
        page = wikipedia.page(title)
        # Preprocess result by converting the HTML content to Markdown to reduce token usage
        page_markdown = markdownify(page.html())
        return f"# URL\n\n{page.url}\n\n# CONTENT\n\n{page_markdown}"
    except wikipedia.exceptions.PageError:
        return f"ERROR: page '{title}' not found."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"ERROR: disambiguation error for '{title}': {e}"
    except Exception as e:
        return f"ERROR: failed to load page '{title}': {e}"


@function_tool
def answer_question(answer: str) -> str:
    """
    End the search process and answer a question. Returns the answer to the question.

    Args:
        answer: The final answer to the user's question.

    Returns:
        The answer (this will be the final output of the agent).
    """
    return answer
