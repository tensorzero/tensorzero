from tensorzero import AsyncTensorZeroGateway, ToolCall, ToolResult
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


async def ask_question(
    t0: AsyncTensorZeroGateway, question: str, verbose: bool = False
):
    """
    Asks a question to the multi-hop retrieval agent and returns the answer.

    Args:
        question (str): The question to ask the agent.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        str: The answer to the question.
    """
    # Initialize the message history with the user's question
    messages = [{"role": "user", "content": question}]

    # The episode ID is used to track the agent's progress (`None` until the first inference)
    episode_id = None

    for _ in range(MAX_INFERENCES):
        print()
        response = await t0.inference(
            function_name="multi_hop_rag_agent",
            input={"messages": messages},
            episode_id=episode_id,
        )

        # Append the assistant's response to the messages
        messages.append({"role": "assistant", "content": response.content})

        # Update the episode ID
        episode_id = response.episode_id

        # Start constructing the tool call results
        output_content_blocks = []

        for content_block in response.content:
            if isinstance(content_block, ToolCall):
                if verbose:
                    print(
                        f"[Tool Call] {content_block.name}: {content_block.arguments}"
                    )

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
                    return content_block.arguments["answer"]
            else:
                # We don't need to do anything with other content blocks.
                print(f"[Other Content Block] {content_block}")

        messages.append({"role": "user", "content": output_content_blocks})
    else:
        # In a production setting, the model could attempt to generate an answer using available information
        # when the search process is stopped; here, we simply throw an exception.
        raise Exception(f"Failed to answer question after {MAX_INFERENCES} inferences.")
