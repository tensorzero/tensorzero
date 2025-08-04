import os
from datetime import datetime

import dotenv
from tensorzero.agents import (
    with_tensorzero_agents_patched,
)
from tools import answer_question, load_wikipedia_page, search_wikipedia, think

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dotenv.load_dotenv()

gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")
print(gateway_url, clickhouse_url)


async def ask_question(question: str, verbose: bool = False) -> str:
    """
    Ask a question to the multi-hop RAG agent using pure Agents SDK.

    Args:
        question: The question to ask
        verbose: Whether to print verbose output

    Returns:
        The agent's answer
    """
    from agents import Agent, ModelSettings, RunConfig, Runner
    from config.generated.schemas.system_schema_model import Model as SystemModel
    from config.generated.schemas.user_schema_model import Model as UserModel

    agent = Agent(
        name="Multi-hop RAG Agent",
        # This is the function name from tensorzero.toml
        model="tensorzero::function_name::multi_hop_rag_agent_openai_v1",
        tools=[think, search_wikipedia, load_wikipedia_page, answer_question],
        model_settings=ModelSettings(extra_body={"tensorzero::episode_id": None}),
    )

    if verbose:
        print(f"\n🤖 Asking agent: {question}")

    # Use the Agents SDK Runner - this handles the entire tool loop automatically
    system_model = SystemModel(date=datetime.now().strftime("%Y-%m-%d"))
    user_model = UserModel(question=question)
    result = await Runner.run(
        agent,
        # TODO:The SDK requires a user message. Maybe it's worth inheriting from the base Runner class to clean up the interface a bit
        "",
        run_config=RunConfig(
            model_settings=ModelSettings(
                metadata=system_model.to_args_dict() | user_model.to_args_dict()
            )
        ),
        max_turns=15,
    )
    # TODO: We need a way of clearing the episode_id after each run. Maybe another advantage of inheriting from the base Runner class
    if verbose:
        print(f"\n✅ Agent response: {result.final_output}")

    return result.final_output


async def main():
    async with with_tensorzero_agents_patched(
        "config/tensorzero.toml", clickhouse_url=clickhouse_url, gateway_url=gateway_url
    ) as tz_context:
        # Test questions from the original example
        questions = [
            "What is a common dish in the hometown of the scientist that won the Nobel Prize for the discovery of the positron?",
            # "What company developed the popular Chinese video game voiced by the same voice actor that voiced a wizard in the anime Konosuba?",
            # "What is the national flower of the country where the mathematician who proved Fermat's Last Theorem was born?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n📋 Question {i}:")
            print(f"   {question}")
            print("-" * 60)

            try:
                answer = await ask_question(question, verbose=True)
                print(f"\n🎯 Final Answer: {answer}")

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"❌ Error: {e}")

            print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
