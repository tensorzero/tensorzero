import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TemperatureToolInput(BaseModel):
    location: str = Field(..., description="The location to get the temperature for.")


class TemperatureTool(BaseTool):
    name: str = "Temperature Tool"
    description: str = "Get the temperature for a given location."
    args_schema: Type[BaseModel] = TemperatureToolInput

    def _run(self, location: str) -> str:
        # TODO: Implement the tool. Let's pretend it's 70 degrees Fahrenheit everywhere.
        return json.dumps(
            {
                "location": location,
                "temperature": 70,
                "unit": "F",
            }
        )
