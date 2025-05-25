import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from minijinja import Environment
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.fine_tuning import FineTuningJob
from tensorzero import AsyncTensorZeroGateway
from tensorzero.internal_optimization_server_types import Sample
from typing_extensions import TypedDict

from ..rendering import get_template_env
from .common import (
    BaseSFTJob,
    FineTuningRequest,
    ValidationError,
    render_message,
    split_validation_data,
    try_template_system,
)
from .openai_analysis import analyze_dataset, get_encoding_for_model

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
else:
    logger.warning("OPENAI_API_KEY environment variable is not set")


def get_openai_client() -> AsyncOpenAI:
    if openai_client is None:
        raise ValueError("OpenAI client is not initialized as credentials are missing")
    return openai_client


class JobInfoOk(TypedDict):
    status: Literal["ok"]
    info: FineTuningJob


class JobInfoError(TypedDict):
    status: Literal["error"]
    message: str
    info: FineTuningJob


JobInfo = Union[JobInfoOk, JobInfoError]


class OpenAISFTJob(BaseSFTJob):
    jobId: str
    jobStatus: str
    fineTunedModel: Optional[str]
    job: JobInfo
    formData: FineTuningRequest
    analysisData: Optional[Dict[str, Any]]  # AnalysisData type

    @staticmethod
    async def from_form_data(
        client: AsyncTensorZeroGateway, data: FineTuningRequest
    ) -> "OpenAISFTJob":
        template_env = get_template_env(client, data.function, data.variant)

        curated_inferences = await client._internal_get_curated_inferences(
            function_name=data.function,
            metric_name=data.metric,
            threshold=data.threshold,
            max_samples=data.maxSamples,
        )

        if not curated_inferences or len(curated_inferences) == 0:
            raise ValueError("No curated inferences found")

        job = await start_sft_openai(
            data.model.name,
            curated_inferences,
            data.validationSplitPercent,
            template_env,
            data,
        )

        return job

    @property
    def jobUrl(self) -> str:
        return f"https://platform.openai.com/finetune/{self.jobId}"

    def status(self) -> Dict[str, Any]:
        if self.jobStatus == "failed":
            job = self.job
            if job["status"] == "error" and "message" in job:
                error = job["message"]
            else:
                error = "Unknown error"
            return {
                "status": "error",
                "modelProvider": "openai",
                "formData": self.formData,
                "jobUrl": self.jobUrl,
                "rawData": self.job,
                "error": error,
            }

        if self.jobStatus == "succeeded":
            if not self.fineTunedModel:
                raise ValueError("Fine-tuned model is undefined")
            return {
                "status": "completed",
                "modelProvider": "openai",
                "formData": self.formData,
                "jobUrl": self.jobUrl,
                "rawData": self.job,
                "result": self.fineTunedModel,
                "analysisData": self.analysisData,
            }

        estimated_completion_time = None
        if self.job["status"] == "ok":
            estimated_completion_time = self.job["info"].estimated_finish

        return {
            "status": "running",
            "modelProvider": "openai",
            "formData": self.formData,
            "rawData": self.job,
            "jobUrl": self.jobUrl,
            "estimatedCompletionTime": estimated_completion_time,
            "analysisData": self.analysisData,
        }

    async def poll(self) -> "OpenAISFTJob":
        if not self.jobId:
            raise ValueError("Job ID is required to poll OpenAI SFT")

        if not openai_client:
            raise ValueError(
                "OpenAI client is not initialized as credentials are missing"
            )

        try:
            job_info = await openai_client.fine_tuning.jobs.retrieve(self.jobId)
        except Exception as error:
            return OpenAISFTJob(
                jobId=self.jobId,
                jobStatus="error",
                fineTunedModel=None,
                job=JobInfoError(
                    status="error", info=self.job["info"], message=str(error)
                ),
                formData=self.formData,
                analysisData=self.analysisData,
            )

        return OpenAISFTJob(
            jobId=job_info.id,
            jobStatus=job_info.status,
            fineTunedModel=job_info.fine_tuned_model,
            job=JobInfoOk(status="ok", info=job_info),
            formData=self.formData,
            analysisData=self.analysisData,
        )


def tensorzero_inference_to_openai_messages(
    sample: Sample, env: Environment
) -> List[ChatCompletionMessageParam]:
    messages = []
    system = try_template_system(sample, env)
    if system:
        messages.append(system)

    for message in sample["input"].get("messages", []):
        for content in message["content"]:
            rendered_message = content_block_to_openai_message(
                content, message["role"], env
            )
            messages.append(rendered_message)

    is_chat_inference = isinstance(sample["output"], list)
    if is_chat_inference:
        output = sample["output"]
        if len(output) != 1:
            raise ValidationError("Chat inference must have exactly one message")
        if output[0]["type"] != "text":
            raise ValidationError("Chat inference must have a text message as output")
        messages.append({"role": "assistant", "content": output[0]["text"]})
    elif "raw" in sample["output"]:
        output = sample["output"]
        messages.append({"role": "assistant", "content": output["raw"]})
    else:
        raise ValidationError("Invalid inference type")

    return messages


def content_block_to_openai_message(
    content: Dict, role: Literal["user", "assistant"], env: Environment
) -> ChatCompletionMessageParam:
    if content["type"] == "text":
        # This looks dumb but PyRight doesn't infer the types correctly without the if/else...?
        if role == "user":
            return {
                "role": role,
                "content": render_message(content, role, env),
            }
        else:
            return {
                "role": role,
                "content": render_message(content, role, env),
            }
    elif content["type"] == "tool_call":
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": content["id"],
                    "type": "function",
                    "function": {
                        "name": content["name"],
                        "arguments": content["arguments"],
                    },
                }
            ],
        }
    elif content["type"] == "tool_result":
        return {
            "role": "tool",
            "tool_call_id": content["id"],
            "content": content["result"],
        }
    elif content["type"] == "file" or content["type"] == "image":
        content_block = (
            content["image"] if content["type"] == "image" else content["file"]
        )
        mime_type = content_block["mime_type"]
        data = content_block["data"]

        if not mime_type.startswith("image/"):
            raise ValidationError(f"Unsupported file mime type: {mime_type}")

        if role != "user":
            raise ValidationError(f"Image content must be user role, found: {role}")

        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{data}",
                    },
                }
            ],
        }
    elif content["type"] == "raw_text":
        # This looks dumb but PyRight doesn't infer the types correctly without the if/else...?
        if role == "user":
            return {
                "role": role,
                "content": content["value"],
            }
        else:
            return {
                "role": role,
                "content": content["value"],
            }
    else:
        raise ValidationError(f"Unsupported content type: {content['type']}")


def validate_and_convert_messages(
    inferences: List[Sample],
    template_env: Environment,
) -> List[List[ChatCompletionMessageParam]]:
    all_messages = []

    for inference in inferences:
        messages = tensorzero_inference_to_openai_messages(inference, template_env)
        all_messages.append(messages)

    return all_messages


async def start_sft_openai(
    model_name: str,
    inferences: List[Sample],
    validation_split_percent: float,
    template_env: Environment,
    request: FineTuningRequest,
) -> "OpenAISFTJob":
    """
    Start a supervised fine-tuning job with OpenAI.

    Args:
        model_name: Name of the OpenAI model to fine-tune
        inferences: List of parsed inference examples
        validation_split_percent: Percentage of data to use for validation
        template_env: Template environment for rendering
        request: Form data containing configuration

    Returns:
        OpenAISFTJob instance
    """
    # Split data into training and validation sets
    train_inferences, val_inferences = split_validation_data(
        inferences, validation_split_percent
    )

    if len(inferences) < 10:
        raise ValueError("Training dataset must have at least 10 examples")

    # Convert inferences to messages for analysis
    train_messages_for_analysis = []
    for inference in train_inferences:
        messages = tensorzero_inference_to_openai_messages(inference, template_env)
        train_messages_for_analysis.append({"messages": messages})

    encoding = get_encoding_for_model(model_name)

    analysis = analyze_dataset(train_messages_for_analysis, model_name, encoding)

    # Prepare analysis data
    first_example_messages: List[ChatCompletionMessageParam] = []

    if train_messages_for_analysis and train_messages_for_analysis[0].get("messages"):
        first_example_messages = [
            {"role": msg.get("role", ""), "content": msg.get("content", "")}
            for msg in train_messages_for_analysis[0]["messages"]
        ]

    analysis_data = {
        "firstExample": first_example_messages,
        "numExamples": len(train_inferences),
        "missingSystemCount": analysis.get("missingSystemCount", 0),
        "missingUserCount": analysis.get("missingUserCount", 0),
        "messageCounts": analysis.get(
            "messageCounts",
            {"min": 0, "max": 0, "mean": 0, "median": 0, "p5": 0, "p95": 0},
        ),
        "tokenCounts": analysis.get(
            "tokenCounts",
            {"min": 0, "max": 0, "mean": 0, "median": 0, "p5": 0, "p95": 0},
        ),
        "assistantTokenCounts": analysis.get(
            "assistantTokenCounts",
            {"min": 0, "max": 0, "mean": 0, "median": 0, "p5": 0, "p95": 0},
        ),
        "tooLongCount": analysis.get("tooLongCount", 0),
        # TODO - port this from typescript
        "tokenLimit": None,
    }

    # Validate and convert messages
    train_messages = validate_and_convert_messages(
        train_inferences,
        template_env,
    )

    val_messages = validate_and_convert_messages(
        val_inferences,
        template_env,
    )

    # Upload examples to OpenAI
    upload_tasks = [
        upload_examples_to_openai(train_messages),
        upload_examples_to_openai(val_messages) if val_messages else None,
    ]

    file_id, val_file_id = await asyncio.gather(
        *[task for task in upload_tasks if task is not None]
    )

    # Create fine-tuning job
    job = await create_openai_fine_tuning_job(model_name, file_id, val_file_id)

    # Create and return OpenAISFTJob
    return OpenAISFTJob(
        jobId=job.id,
        jobStatus="created",
        fineTunedModel=None,
        job={"status": "ok", "info": job},
        formData=request,
        analysisData=analysis_data,
    )


async def upload_examples_to_openai(
    samples: List[List[ChatCompletionMessageParam]],
) -> str:
    """
    Uploads examples to OpenAI by creating a file.

    Args:
        samples: List of message lists to upload

    Returns:
        File ID from OpenAI
    """
    # Convert samples to JSONL format
    jsonl = "\n".join(
        [json.dumps({"messages": messages}) for messages in samples]
    ).encode("utf-8")
    openai_file = await get_openai_client().files.create(
        file=jsonl, purpose="fine-tune"
    )
    return openai_file.id


async def create_openai_fine_tuning_job(
    model: str, train_file_id: str, val_file_id: Optional[str] = None
) -> FineTuningJob:
    """
    Create a fine-tuning job with OpenAI.

    Args:
        model: The model to fine-tune
        train_file_id: ID of the training file
        val_file_id: Optional ID of the validation file

    Returns:
        Job data from the OpenAI API
    """

    try:
        # Create the fine-tuning job
        print("Creating fine-tuning job with OpenAI...")
        job = await get_openai_client().fine_tuning.jobs.create(
            model=model, training_file=train_file_id, validation_file=val_file_id
        )
        print("Fine-tuning job created successfully: ", job)
        return job
    except Exception as error:
        print(f"Error creating fine-tuning job: {error}")
        raise
