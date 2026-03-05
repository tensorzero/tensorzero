import json
from asyncio import Semaphore
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tools import load_wikipedia_page, parse_tool_arguments, search_wikipedia

# ## Agentic RAG
#
# Here we define the function that will be used to ask a question to the multi-hop retrieval agent.
#
# The function takes a question and launches a multi-hop retrieval process.
# The agent will make a number of tool calls to search for information and answer the question.
#
# The function will return the answer to the question.

# The maximum number of inferences the agent will make.
MAX_INFERENCES = 20

# The maximum number of characters in the messages before compacting.
MAX_MESSAGE_LENGTH = 100_000


@dataclass
class RunResult:
    answer: str
    t: int


async def ask_question(
    openai_client: AsyncOpenAI,
    semaphore: Semaphore,
    question: str,
    episode_id: str,
    verbose: bool = False,
) -> RunResult:
    """
    Asks a question to the multi-hop retrieval agent and returns the answer.

    Args:
        openai_client: The OpenAI client pointed at the TensorZero Gateway.
        semaphore: A semaphore to limit concurrency.
        question: The question to ask the agent.
        episode_id: The episode ID for the workflow evaluation run.
        verbose: Whether to print verbose output. Defaults to False.

    Returns:
        RunResult: The answer and number of iterations.
    """
    # Initialize the message history with the user's question
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": question}]

    t = None
    for t in range(MAX_INFERENCES):
        if verbose:
            print()
        async with semaphore:
            response = await openai_client.chat.completions.create(
                model="tensorzero::function_name::multi_hop_rag_agent",
                messages=messages,
                extra_body={"tensorzero::episode_id": episode_id},
            )

        assistant_message = response.choices[0].message

        # Append the assistant's response to the messages
        messages.append(assistant_message)  # type: ignore

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            # No tool calls — the model responded with text only
            if verbose:
                print(f"[Text Response] {assistant_message.content}")
            continue

        # Process each tool call
        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = parse_tool_arguments(tool_call.function.arguments)

            if verbose:
                print(f"[Tool Call] {name}: {arguments}")

            if name == "search_wikipedia":
                result = search_wikipedia(arguments)
            elif name == "load_wikipedia_page":
                result = load_wikipedia_page(arguments)
            elif name == "think":
                # The `think` tool is just used to plan the next steps, and there's no actual tool to call.
                # Some providers like OpenAI require a tool result, so we'll provide an empty string.
                result = ""
            elif name == "answer_question":
                return RunResult(answer=arguments.get("answer", ""), t=t)
            else:
                result = f"ERROR: unknown tool `{name}`"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        approx_message_length = len(str(messages))
        if approx_message_length > MAX_MESSAGE_LENGTH:
            try:
                messages = await compact_context(openai_client, semaphore, question, messages, episode_id, verbose)
            except Exception as e:
                print(f"Error compacting context: {e}")
                messages = messages[:-2]
    else:
        if t is None:
            raise ValueError("`MAX_INFERENCES` must be positive.")

        # In a production setting, the model could attempt to generate an answer using available information
        # when the search process is stopped; here, we simply return a failure message.
        return RunResult(answer="The agent failed to answer the question.", t=t)


async def compact_context(
    openai_client: AsyncOpenAI,
    semaphore: Semaphore,
    question: str,
    messages: list[ChatCompletionMessageParam],
    episode_id: str,
    verbose: bool = False,
) -> list[ChatCompletionMessageParam]:
    if verbose:
        print("Compacting context...")
    async with semaphore:
        response = await openai_client.chat.completions.create(
            model="tensorzero::function_name::compact_context",
            messages=[
                {
                    "role": "system",
                    "content": [  # type: ignore
                        {
                            "type": "text",
                            "tensorzero::arguments": {"question": question},
                        }
                    ],
                },
                *messages,
            ],
            extra_body={"tensorzero::episode_id": episode_id},
        )

    compacted_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response.choices[0].message.content or ""},
    ]
    return compacted_messages
