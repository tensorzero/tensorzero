import json
from typing import Annotated

from ag2 import AssistantAgent, UserProxyAgent


def main():
    llm_config = {
        "config_list": [
            {
                "model": "tensorzero::model_name::openai::gpt-4o-mini",
                "base_url": "http://localhost:3000/openai/v1",
                "api_key": "not-needed",
            }
        ],
    }

    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=(
            "You are a helpful assistant. "
            "You can look up the current temperature for any location. "
            "Reply TERMINATE when the task is done."
        ),
    )

    user = UserProxyAgent(
        name="user",
        human_input_mode="ALWAYS",
        code_execution_config=False,
    )

    @user.register_for_execution()
    @assistant.register_for_llm(
        description="Get the current temperature for a given location."
    )
    def temperature_api(
        location: Annotated[str, "The location to get the temperature for"],
    ) -> str:
        # Pretend it's 25 degrees Celsius everywhere!
        return json.dumps({"location": location, "temperature": 25, "unit": "C"})

    user.initiate_chat(
        assistant,
        message="What is the weather in New York City?",
    )


if __name__ == "__main__":
    main()
