"""
TensorZero + OpenAI Agents SDK Integration Example

This example shows how to use the simple agentic RAG functionality with the
OpenAI Agents SDK integration, leveraging the existing TensorZero configuration.

Install: pip install tensorzero[agents]
"""

import asyncio
import wikipedia
from markdownify import markdownify
from agents import Agent, Runner, function_tool

# Import TensorZero agents integration
import tensorzero.agents as tz_agents


# Define tools using OpenAI Agents SDK decorators
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
    print(f"🤔 Thinking: {thought}")
    return ""


@function_tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for articles related to your query.

    Args:
        query: The search query to find relevant Wikipedia articles.

    Returns:
        A list of Wikipedia article titles that match your search.
    """
    print(f"🔍 Searching Wikipedia for: {query}")
    try:
        results = wikipedia.search(query, results=5)
        if results:
            return f"Found Wikipedia articles: {', '.join(results)}"
        else:
            return f"No Wikipedia articles found for: {query}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@function_tool
def load_wikipedia_page(title: str) -> str:
    """
    Load the full content of a specific Wikipedia page.

    Args:
        title: The exact title of the Wikipedia page to load.

    Returns:
        The full text content of the Wikipedia page in markdown format.
    """
    print(f"📖 Loading Wikipedia page: {title}")
    try:
        page = wikipedia.page(title)
        # Convert HTML to markdown for better readability
        content = markdownify(page.content)
        return f"# {page.title}\n\n{content}"
    except wikipedia.exceptions.DisambiguationError as e:
        return (
            f"Disambiguation needed for '{title}'. Options: {', '.join(e.options[:5])}"
        )
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page not found: {title}"
    except Exception as e:
        return f"Error loading Wikipedia page: {str(e)}"


@function_tool
def answer_question(answer: str) -> str:
    """
    Provide your final answer to the user's question.
    Use this tool only when you have gathered sufficient information to give a complete answer.

    Args:
        answer: Your complete answer to the user's original question.

    Returns:
        Confirmation that the answer was provided.
    """
    print(f"✅ Final Answer: {answer}")
    return "Answer provided to user."


async def create_tensorzero_integrated_agent() -> Agent:
    """
    Create an agent using TensorZero configuration with Agents SDK integration.

    This automatically:
    1. Loads the TensorZero configuration from tensorzero.toml
    2. Detects templated functions and their variables
    3. Sets up OpenAI client patching to route through TensorZero
    4. Enables automatic template variable handling
    """

    # Set up TensorZero + Agents SDK integration
    config = await tz_agents.setup_tensorzero_agents(
        config_path="config/tensorzero.toml",
        base_url="http://localhost:3000",  # TensorZero gateway
        api_key="your-api-key",
    )

    # Inspect what was loaded from TensorZero config
    print(f"\n📋 TensorZero Configuration Loaded:")
    print(f"   Templated Functions: {list(config.templated_functions.keys())}")
    print(f"   Available Tools: {list(config.available_tools.keys())}")

    # Create agent using TensorZero function
    # This automatically inherits the template from TensorZero config
    agent = tz_agents.create_agent_from_tensorzero_function(
        function_name="multi_hop_rag_agent",
        variant_name="baseline",
        agent_name="TensorZero RAG Agent",
        tools=[think, search_wikipedia, load_wikipedia_page, answer_question],
    )

    return agent


async def create_manual_agent() -> Agent:
    """
    Create an agent manually specifying the model name.

    This shows how you can still use TensorZero models directly
    while getting all the benefits of automatic templating.
    """

    # Set up TensorZero integration first
    await tz_agents.setup_tensorzero_agents(
        config_path="config/tensorzero.toml",
        base_url="http://localhost:3000",
        api_key="your-api-key",
    )

    # Create agent with explicit TensorZero model reference
    # The system prompt template is automatically loaded from TensorZero config
    agent = Agent(
        name="Manual TensorZero RAG Agent",
        model="tensorzero::function_name::multi_hop_rag_agent::baseline",
        instructions="",  # Will be overridden by TensorZero template
        tools=[think, search_wikipedia, load_wikipedia_page, answer_question],
    )

    return agent


async def ask_question_with_integration(question: str, use_auto_agent: bool = True):
    """
    Ask a question using the TensorZero + Agents SDK integration.

    Args:
        question: The question to ask
        use_auto_agent: If True, use the auto-created agent. If False, use manual agent.
    """
    print(f"❓ Question: {question}")
    print("=" * 80)

    if use_auto_agent:
        agent = await create_tensorzero_integrated_agent()
        print("🤖 Using auto-created agent from TensorZero function")
    else:
        agent = await create_manual_agent()
        print("🔧 Using manually configured agent")

    # Run the agent with the question
    # All TensorZero features work automatically:
    # - A/B testing between variants
    # - Observability and logging to ClickHouse
    # - Template variable resolution
    # - Metrics collection
    runner = Runner(agent=agent)

    print(f"\n🚀 Starting conversation...")
    response = await runner.run(question)

    print(f"\n📊 Integration Benefits Automatic:")
    print(f"   ✅ TensorZero observability and logging")
    print(f"   ✅ A/B testing if multiple variants configured")
    print(f"   ✅ Template variable handling")
    print(f"   ✅ All TensorZero gateway features")

    return response


async def compare_implementations():
    """Compare the pure Agents SDK approach vs TensorZero integration."""

    question = "What is a common dish in the hometown of the scientist who won the Nobel Prize for the discovery of the positron?"

    print("🧪 COMPARISON: Pure Agents SDK vs TensorZero Integration")
    print("=" * 80)

    print("\n1️⃣ TensorZero Integration (Recommended)")
    print("-" * 50)
    response_tz = await ask_question_with_integration(question, use_auto_agent=True)

    print(f"\n🎯 Key Benefits of TensorZero Integration:")
    print(f"   📊 Automatic observability - all interactions logged to ClickHouse")
    print(f"   🧪 A/B testing - can easily test different prompts/models")
    print(f"   🎨 Template system - prompts managed centrally in config")
    print(f"   🚀 Production ready - proven scaling and reliability")
    print(f"   📈 Analytics - built-in metrics and monitoring")

    return response_tz


async def main():
    """Main function demonstrating the TensorZero + Agents SDK integration."""

    print("🎉 TensorZero + OpenAI Agents SDK Integration Demo")
    print("=" * 80)

    # Run comparison
    await compare_implementations()

    print(f"\n✨ Integration Summary:")
    print(
        f"   • One-line setup: await tz_agents.setup_tensorzero_agents('config.toml')"
    )
    print(f"   • Automatic template handling from TensorZero configuration")
    print(f"   • All TensorZero production features work transparently")
    print(f"   • Clean Agents SDK code with enterprise-grade infrastructure")


if __name__ == "__main__":
    asyncio.run(main())
