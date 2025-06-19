from typing import List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, llm, task

from example.tools.weather_tool import TemperatureTool


@CrewBase
class Example:
    agents: List[BaseAgent]
    tasks: List[Task]

    @llm
    def tensorzero(self, model: str) -> LLM:
        # IMPORTANT: We need to add the `openai/` prefix to the CrewAI to use the OpenAI API format
        # `model` should be `tensorzero::model_name::your_model_name` or `tensorzero::function_name::your_function_name`

        return LLM(
            model=f"openai/{model}",
            base_url="http://localhost:3000/openai/v1",  # point to your TensorZero gateway
            api_key="dummy",  # credentials live inside the TensorZero gateway
            # Any other LLM parameters can be added here...
            temperature=0.0,
        )

    @agent
    def weather_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["weather_analyst"],  # type: ignore[index]
            llm=self.tensorzero("tensorzero::model_name::openai::gpt-4o-mini"),
            tools=[TemperatureTool()],
        )

    @task
    def report_weather(self) -> Task:
        return Task(
            config=self.tasks_config["report_weather"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
