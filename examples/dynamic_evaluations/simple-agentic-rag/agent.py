from asyncio import Semaphore
from dataclasses import dataclass

from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    Message,
    ToolCall,
    ToolResult,
)
from tensorzero.util import UUID
from tools import load_wikipedia_page, search_wikipedia

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
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: str,
    episode_id: UUID,
    verbose: bool = False,
) -> RunResult:
    """
    Asks a question to the multi-hop retrieval agent and returns the answer.

    Args:
        question (str): The question to ask the agent.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        str: The answer to the question.
    """
    # Initialize the message history with the user's question
    messages: list[Message] = [{"role": "user", "content": question}]

    t = None
    for t in range(MAX_INFERENCES):
        if verbose:
            print()
        async with semaphore:
            response = await t0.inference(
                function_name="multi_hop_rag_agent",
                input={"messages": messages},
                episode_id=episode_id,
            )
            assert isinstance(response, ChatInferenceResponse)

        # Append the assistant's response to the messages
        messages.append({"role": "assistant", "content": response.content})

        # Start constructing the tool call results
        output_content_blocks = []

        for content_block in response.content:
            if isinstance(content_block, ToolCall):
                if verbose:
                    print(f"[Tool Call] {content_block.name}: {content_block.arguments}")

                if content_block.name is None or content_block.arguments is None:
                    output_content_blocks.append(
                        ToolResult(
                            name=content_block.raw_name,
                            id=content_block.id,
                            result="ERROR: invalid tool call",
                        )
                    )
                elif content_block.name == "search_wikipedia":
                    output_content_blocks.append(search_wikipedia(content_block))
                elif content_block.name == "load_wikipedia_page":
                    output_content_blocks.append(load_wikipedia_page(content_block))
                elif content_block.name == "think":
                    # The `think` tool is just used to plan the next steps, and there's no actual tool to call.
                    # Some providers like OpenAI require a tool result, so we'll provide an empty string.
                    output_content_blocks.append(
                        ToolResult(
                            name="think",
                            id=content_block.id,
                            result="",
                        )
                    )
                elif content_block.name == "answer_question":
                    return RunResult(answer=content_block.arguments["answer"], t=t)
            else:
                # We don't need to do anything with other content blocks.
                print(f"[Other Content Block] {content_block}")

        messages.append({"role": "user", "content": output_content_blocks})
        approx_message_length = len(str(messages))
        if approx_message_length > MAX_MESSAGE_LENGTH:
            try:
                messages = await compact_context(t0, semaphore, question, messages, episode_id, verbose)
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
    t0: AsyncTensorZeroGateway,
    semaphore: Semaphore,
    question: str,
    messages: list[Message],
    episode_id: UUID,
    verbose: bool = False,
):
    if verbose:
        print("Compacting context...")
    async with semaphore:
        response = await t0.inference(
            function_name="compact_context",
            input={
                "system": {"question": question},
                "messages": messages,
            },
            episode_id=episode_id,
        )
        assert isinstance(response, ChatInferenceResponse)

    compacted_messages: list[Message] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response.content},
    ]
    return compacted_messages
