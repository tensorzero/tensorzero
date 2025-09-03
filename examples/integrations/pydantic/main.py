from pydantic import BaseModel, Field
from tensorzero import TensorZeroGateway


class Email(BaseModel):
    email: str = Field(description="extracted email")


with TensorZeroGateway.build_embedded(config_file="config/tensorzero.toml") as client:
    result = client.inference(
        function_name="extract_email",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Please contact us at hello@tensorzero.com for further inquiries.",
                }
            ]
        },
        output_schema=Email.model_json_schema(),
    )

email = Email.model_validate(result.output.parsed)

print(email.model_dump_json())
