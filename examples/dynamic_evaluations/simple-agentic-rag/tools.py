import wikipedia
from markdownify import markdownify
from tensorzero import ToolCall, ToolResult

# ## Tools
#
# We define as Python functions the tools that will be used by the model.
#
# Here, we have a tool for searching Wikipedia and a tool for loading a Wikipedia page.
# These are also described in the config/tensorzero.toml file so that our agent automatically knows about them.


def search_wikipedia(tool_call: ToolCall) -> ToolResult:
    """
    Searches Wikipedia for a given query and returns a list of search results.

    Args:
        tool_call (ToolCall): A tool call object containing the search query in its arguments.
            Expected arguments: {"query": str}

    Returns:
        ToolResult: A tool result containing the newline-separated list of Wikipedia search results.
            The result field contains the search results as a string.
    """
    if tool_call.arguments is None:
        raise ValueError("The tool call doesn't have `arguments`, so it must not have parsed correctly.")

    query = tool_call.arguments.get("query")
    if query is None:
        raise ValueError("The `query` argument is required for `search_wikipedia` tool.")

    search_wikipedia_result = "\n".join(wikipedia.search(query))

    return ToolResult(
        name="search_wikipedia",
        id=tool_call.id,
        result=search_wikipedia_result,
    )


def load_wikipedia_page(tool_call: ToolCall) -> ToolResult:
    """
    Loads and formats the content of a Wikipedia page.

    Args:
        tool_call (ToolCall): A tool call object containing the page title in its arguments.
            Expected arguments: {"title": str}

    Returns:
        ToolResult: A tool result containing the formatted Wikipedia page content.
            The result field contains the page URL and content in Markdown format.
            If the page is not found or there's a disambiguation error, returns an error message.
    """
    if tool_call.arguments is None:
        raise ValueError("The tool call doesn't have `arguments`, so it must not have parsed correctly.")

    title = tool_call.arguments.get("title")
    if title is None:
        raise ValueError("The `title` argument is required for `load_wikipedia_page` tool.")

    try:
        page = wikipedia.page(title)
        # Preprocess result by converting the HTML content to Markdown to reduce token usage
        page_markdown = markdownify(page.html())
        load_wikipedia_page_result = f"# URL\n\n{page.url}\n\n# CONTENT\n\n{page_markdown}"
    except wikipedia.exceptions.PageError:
        load_wikipedia_page_result = f"ERROR: page '{title}' not found."
    except wikipedia.exceptions.DisambiguationError as e:
        load_wikipedia_page_result = f"ERROR: disambiguation error for '{title}': {e}"

    return ToolResult(
        name="load_wikipedia_page",
        id=tool_call.id,
        result=load_wikipedia_page_result,
    )
