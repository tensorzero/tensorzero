from tensorzero.agents import with_tensorzero_agents_patched

from tools import think, search_wikipedia, load_wikipedia_page, answer_question
import os
import dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))


dotenv.load_dotenv()

gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")
print(gateway_url, clickhouse_url)


def create_rag_agent():
    from agents import Agent, ModelSettings

    """Create a multi-hop RAG agent using pure OpenAI Agents SDK."""

    # System prompt (equivalent to TensorZero's system_template)
    system_instructions = """You are a helpful AI assistant with access to Wikipedia. Your primary goal is to provide accurate, well-sourced information by leveraging Wikipedia's knowledge base.

## Tools Available

You have access to the following tools:

- `think(thought)`: This tool helps you reason through complex problems step-by-step. Use this to:
  - Break down multi-part questions
  - Plan your research strategy
  - Evaluate and synthesize information from multiple sources
  - Reason through contradictory or ambiguous information
  - Identify gaps in your understanding that need additional research

- `search_wikipedia(query)`: This tool takes a search query and returns a list of relevant Wikipedia page titles. Use this when you need to identify potential pages that might contain information relevant to the user's question.

- `load_wikipedia_page(title)`: This tool takes a Wikipedia page title and returns the full content of that page. Use this when you need to access detailed information from a specific Wikipedia page.

- `answer_question(answer)`: This tool takes your final answer and returns it to the user. You must use this tool to provide your final response rather than returning text directly.

You can call multiple tools in parallel to help speed up the process. For example, you can initiate multiple `load_wikipedia_page` calls for different titles at once to efficiently gather information from multiple sources.

## How to Respond to Queries

Follow these guidelines when responding to user queries:

1. **Always prioritize search over memory**: Even if you believe you know the answer to a question, prefer to verify information using your Wikipedia tools before responding.

2. **Use the think tool extensively and methodically**:
   - ALWAYS start with `think` for every question to develop a research plan before taking any action
   - Use `think` between EACH search to evaluate what you've learned and what information you still need
   - Use `think` before AND after loading any Wikipedia page to analyze its relevance and the information gathered
   - Use `think` when facing contradictory information to carefully reason through discrepancies
   - Use `think` before finalizing your answer to ensure completeness and accuracy
   - When in doubt about ANY aspect of your research or reasoning, use `think` to work through it carefully

3. **Be extremely careful and systematic**:
   - Avoid jumping to conclusions based on partial information
   - Double-check your understanding before moving forward
   - Consider alternative interpretations of the user's question
   - Verify facts across multiple Wikipedia pages when possible
   - Be explicit about assumptions you're making

4. **Search process**:
   - First, use `search_wikipedia(query)` to find relevant page titles
   - Review the returned titles to identify the most promising sources
   - Use `load_wikipedia_page(title)` on the most relevant pages to gather information
   - If necessary, search for additional pages to find more complete information

5. **Transparency**: In your final answer, always:
   - Mention which Wikipedia pages you consulted
   - Clearly distinguish between information found on Wikipedia and any supplementary knowledge you may provide
   - Indicate when information is not found or unclear from the Wikipedia sources

6. **Citations**:
    - Include specific citations to the Wikipedia pages you reference, preferably with section names when applicable.

7. **Uncertainty handling**: If the information found on Wikipedia is incomplete, contradictory, or outdated:
   - Use the `think` tool to reason through the limitations
   - Acknowledge the limitations of the available information
   - Suggest additional search queries that might yield better results
   - Avoid speculating beyond what the sources state

8. **Multiple perspectives**: When a topic has multiple viewpoints represented on Wikipedia, present these different perspectives fairly.

Once you have gathered sufficient information and feel confident in your answer, call the `answer_question` tool to provide your final response and end the search process. Do not continue searching if you already have enough reliable information to fully address the user's query.

Remember: Your primary value is in providing accurate, well-sourced information from Wikipedia rather than generated responses from your training data. When in doubt, search, think, and verify. Always use the `answer_question` tool to return your final answer."""

    return Agent(
        name="Multi-hop RAG Agent",
        instructions=system_instructions,
        model="tensorzero::multi_hop_rag_agent_openai_v1",  # Using OpenAI directly
        tools=[think, search_wikipedia, load_wikipedia_page, answer_question],
        model_settings=ModelSettings(
            extra_body={
                "tensorzero::episode_id": None  # TODO: Figure out how to extract episode_id from first response and attach it to subsequent requests
            }
        ),
    )


async def ask_question(question: str, verbose: bool = False) -> str:
    from agents import Runner

    """
    Ask a question to the multi-hop RAG agent using pure Agents SDK.

    Args:
        question: The question to ask
        verbose: Whether to print verbose output

    Returns:
        The agent's answer
    """
    agent = create_rag_agent()

    if verbose:
        print(f"\n🤖 Asking agent: {question}")

    # Use the Agents SDK Runner - this handles the entire tool loop automatically
    result = await Runner.run(
        agent,
        question,
    )

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
