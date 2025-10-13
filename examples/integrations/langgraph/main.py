from uuid import UUID

from langgraph.graph import END, StateGraph
from tensorzero import (
    ChatInferenceResponse,
    Message,
    TensorZeroGateway,
    Text,
    ToolCall,
    ToolResult,
)
from typing_extensions import TypedDict

# Initialize TensorZero Gateway
t0 = TensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml",
)


class State(TypedDict):
    """
    Represents the state of the agent.

    For this example, we use a simple state that includes a list of messages and a TensorZero episode ID.
    """

    messages: list  # message history
    episode_id: UUID  # TensorZero episode ID


class TemperatureAPI:
    """
    A tool for a (fake) temperature API.

    Our agent can use this tool to get the current temperature for a given location.
    """

    def __call__(self, state: State):
        tool_call_id = None

        # Check if we received a tool call for `temperature_api`
        for content_block in state["messages"][-1]["content"]:
            if isinstance(content_block, ToolCall) and content_block.name == "temperature_api":
                tool_call_id = content_block.id
                break

        if tool_call_id is None:
            raise ValueError("TemperatureAPI didn't find a tool call for `temperature_api`")

        # Pretend it's 25 degrees Celsius everywhere!
        message = Message(
            role="user",
            content=[ToolResult(name="temperature_api", result="25", id=tool_call_id)],
        )

        return {"messages": state["messages"] + [message]}


def route_chatbot_response(state: State):
    """
    Route execution based on the chatbot's message content.

    If the last message contains a tool call for `temperature_api`, route to the "temperature_api" node.
    Otherwise, finish running the LangGraph graph.
    """
    for content_block in state["messages"][-1]["content"]:
        if isinstance(content_block, ToolCall):
            if content_block.name == "temperature_api":
                return "temperature_api"
            else:
                raise ValueError(f"Unknown Tool Call: {content_block.raw_name}")

    return END


def chatbot(state: State):
    """
    Call the `chatbot` function in TensorZero.
    """
    response = t0.inference(
        function_name="chatbot",
        episode_id=state.get("episode_id"),
        input={"messages": state["messages"]},
    )

    if not isinstance(response, ChatInferenceResponse):
        raise ValueError(f"Unexpected Response Type: {type(response)}")

    assistant_message = Message(role="assistant", content=response.content)

    return {
        "messages": state["messages"] + [assistant_message],
        "episode_id": response.episode_id,
    }


def build_graph():
    """
    Build the LangGraph graph for the agent.
    """
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("temperature_api", TemperatureAPI())

    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", route_chatbot_response)
    graph_builder.add_edge("temperature_api", "chatbot")

    graph = graph_builder.compile()

    print(graph.get_graph().draw_ascii())

    return graph


def main():
    graph = build_graph()

    # Set the initial state for the agent
    messages = []
    episode_id = None

    while True:
        # Collect user input
        user_input = input("\n[User]\n")
        user_message = Message(role="user", content=user_input)
        messages.append(user_message)

        # Process the user input
        for event in graph.stream({"messages": messages, "episode_id": episode_id}):
            if "chatbot" in event:
                messages = event["chatbot"]["messages"]
                episode_id = event["chatbot"]["episode_id"]

                for content_block in event["chatbot"]["messages"][-1]["content"]:
                    if isinstance(content_block, Text):
                        print("\n[Assistant]")
                        print(content_block.text)
                    elif isinstance(content_block, ToolCall):
                        print(f"\n[Tool Call: {content_block.raw_name}]")
                        print(content_block.raw_arguments)
                    else:
                        raise NotImplementedError(f"Unknown Content Block: {content_block}")

            if "temperature_api" in event:
                messages = event["temperature_api"]["messages"]
                print("\n[Tool Result: temperature_api]")
                print(messages[-1]["content"][-1].result)


if __name__ == "__main__":
    main()
