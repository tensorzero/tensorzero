"""
This module handles supervised fine-tuning and deployment of models using the Fireworks AI API.

The high-level flow is:
1. User submits fine-tuning job configuration via the UI form (SFTFormValues)
2. FireworksSFTJob class is instantiated with the form data
3. Curated training data is retrieved from ClickHouse based on:
   - Selected function
   - Selected metric
   - Max samples limit
4. Training data is formatted according to Fireworks API requirements
5. Fine-tuning job is launched via Fireworks API endpoints (we get a job ID here)
6. Job status is polled periodically to track progress
7. Once complete, the fine-tuned model ID is stored
8. The fine-tuned model is then deployed-- this also needs to be polled
9. Once this is completed we have a path we can use for inference

The FireworksSFTJob class extends the base SFTJob class to provide
Fireworks-specific implementation of job creation, status polling,
result handling, and deployment management.
"""

import asyncio
import json
import logging
import os
import typing as t
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from minijinja import Environment
from pydantic import BaseModel, ConfigDict
from tensorzero import AsyncTensorZeroGateway
from tensorzero.internal_optimization_server_types import Sample
from typing_extensions import TypedDict
from uuid_utils import uuid7

from ..rendering import get_template_env

# Import modules from equivalent Python packages
from .common import BaseSFTJob, FineTuningRequest, render_message, try_template_system

logger = logging.getLogger(__name__)


# Retrieves the API key for the Fireworks API from environment variables
# Logs a warning if the key is not set
def get_api_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("WARNING: FIREWORKS_API_KEY is not set")
        return ""
    return key


FIREWORKS_API_KEY = get_api_key()

# Base URL for the Fireworks API
FIREWORKS_API_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
FIREWORKS_CLIENT = httpx.AsyncClient(
    base_url=FIREWORKS_API_URL,
    headers={
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    },
)


# Retrieves the account ID for the Fireworks API from environment variables
# Logs a warning if the ID is not set
def get_account_id():
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        print("WARNING: FIREWORKS_ACCOUNT_ID is not set")
        return ""
    return account_id


FIREWORKS_ACCOUNT_ID = get_account_id()


class JobInfo(TypedDict):
    status: Literal["ok", "error"]
    info: Union[str, Dict[str, Any]]
    message: t.Optional[str]


class FireworksSFTJobParams(TypedDict):
    jobPath: str
    jobStatus: str
    jobId: str
    modelId: t.Optional[str]
    modelPath: t.Optional[str]
    jobInfo: JobInfo
    formData: Dict[str, Any]


class FineTuningJobStatus(str, Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    CREATING = "CREATING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETING = "DELETING"


class FineTuningJobResponse(BaseModel):
    state: FineTuningJobStatus
    modelId: Optional[str] = None
    baseModel: Optional[str] = None
    batchSize: Optional[int] = None
    createTime: Optional[str] = None
    createdBy: Optional[str] = None
    dataset: Optional[str] = None
    evaluationSplit: Optional[float] = None
    fineTuningJobId: Optional[str] = None
    fineTuningJobName: Optional[str] = None
    fineTuningJobPath: Optional[str] = None
    evaluation: Optional[bool] = None
    evaluationDataset: Optional[str] = None
    learningRate: Optional[float] = None
    loraRank: Optional[int] = None
    loraTargetModules: Optional[List[str]] = None
    maskToken: Optional[str] = None
    microBatchSize: Optional[int] = None
    name: Optional[str] = None
    padToken: Optional[str] = None
    status: Optional[Dict[str, Optional[str]]] = None


class FireworksMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class FireworksExample(TypedDict):
    messages: List[FireworksMessage]


class FireworksSFTJob(BaseSFTJob):
    """Handles supervised fine-tuning jobs with Fireworks AI"""

    jobPath: str
    jobStatus: str
    jobId: str
    modelId: Optional[str]
    modelPath: Optional[str]
    jobInfo: JobInfo
    formData: Dict[str, Any]

    model_config = ConfigDict(extra="allow")

    @staticmethod
    async def from_form_data(
        client: AsyncTensorZeroGateway, data: FineTuningRequest
    ) -> "FireworksSFTJob":
        # TODO - move this out of 'from_form_data'
        template_env = get_template_env(client, data.function, data.variant)

        curated_inferences = await client._internal_get_curated_inferences(
            function_name=data.function,
            metric_name=data.metric,
            threshold=data.threshold,
            max_samples=data.maxSamples,
        )

        if not curated_inferences or len(curated_inferences) == 0:
            raise ValueError("No curated inferences found")

        try:
            job = await start_sft_fireworks(
                data.model.name,
                curated_inferences,
                data.validationSplitPercent,
                template_env,
                data,
            )
        except Exception:
            raise
            # raise ValueError(f"Failed to start Fireworks SFT job: {str(error)}")

        return FireworksSFTJob(
            jobPath=job["name"],
            jobStatus="RUNNING",
            jobId=data.jobId,
            modelId=None,
            modelPath=None,
            jobInfo=JobInfo(
                status="ok",
                info=job,
                message=None,
            ),
            formData=data.model_dump(),
        )

    @property
    def job_url(self) -> str:
        """Get the URL for viewing the job in the Fireworks dashboard"""
        job_id = self.jobPath.split("/")[-1]
        if not job_id:
            raise ValueError("Failed to parse job ID from path")
        return f"https://fireworks.ai/dashboard/fine-tuning/v1/{job_id}"

    def status(self) -> Dict[str, Any]:  # SFTJobStatus equivalent
        """Get the current status of the job"""
        if self.jobStatus == "FAILED":
            error = (
                self.jobInfo.get("message", "Unknown error")
                if self.jobInfo["status"] == "error"
                else "Unknown error"
            )
            return {
                "status": "error",
                "modelProvider": "fireworks",
                "formData": self.formData,
                "jobUrl": self.job_url,
                "rawData": self.jobInfo,
                "error": error,
            }

        if self.jobStatus == "DEPLOYED":
            if not self.modelPath:
                raise ValueError("Model path is undefined for deployed job")
            return {
                "status": "completed",
                "modelProvider": "fireworks",
                "formData": self.formData,
                "jobUrl": self.job_url,
                "result": self.modelPath,
                "rawData": self.jobInfo,
            }

        return {
            "status": "running",
            "modelProvider": "fireworks",
            "formData": self.formData,
            "jobUrl": self.job_url,
            "rawData": self.jobInfo,
        }

    async def poll(self) -> "FireworksSFTJob":
        """Poll for updates to the job status"""
        try:
            if not self.modelId:
                # If we don't have a model ID, training is still running so we need to poll for it
                job_info = await get_fine_tuning_job_details(self.jobPath)
                status = job_info.state

                if status == FineTuningJobStatus.COMPLETED:
                    model_id = job_info.modelId
                    if not model_id:
                        raise ValueError("Model ID not found after job completed")

                    # Begin deployment process
                    await deploy_model_request(FIREWORKS_ACCOUNT_ID, model_id)

                    return FireworksSFTJob(
                        jobPath=self.jobPath,
                        jobStatus="DEPLOYING",
                        jobId=self.jobId,
                        modelId=model_id,
                        modelPath=None,
                        jobInfo=JobInfo(
                            status="ok", info=job_info.model_dump(), message=None
                        ),
                        formData=self.formData,
                    )
                else:
                    return FireworksSFTJob(
                        jobPath=self.jobPath,
                        jobStatus="TRAINING",
                        jobId=self.jobId,
                        modelId=None,
                        modelPath=None,
                        jobInfo=JobInfo(
                            status="ok",
                            info=job_info.model_dump(),
                            message=None,
                        ),
                        formData=self.formData,
                    )
            else:
                # If we do have a model ID, we need to poll for the deployment
                deploy_model_response = await deploy_model_request(
                    FIREWORKS_ACCOUNT_ID, self.modelId
                )
                status = get_deployment_status(deploy_model_response)

                if status == "DEPLOYED":
                    model_path = (
                        f"accounts/{FIREWORKS_ACCOUNT_ID}/models/{self.modelId}"
                    )

                    return FireworksSFTJob(
                        jobPath=self.jobPath,
                        jobStatus="DEPLOYED",
                        jobId=self.jobId,
                        modelId=self.modelId,
                        modelPath=model_path,
                        jobInfo=JobInfo(
                            status="ok", info=deploy_model_response, message=None
                        ),
                        formData=self.formData,
                    )
                else:
                    return FireworksSFTJob(
                        jobPath=self.jobPath,
                        jobStatus="DEPLOYING",
                        jobId=self.jobId,
                        modelId=self.modelId,
                        modelPath=None,
                        jobInfo=JobInfo(
                            status="ok", info=deploy_model_response, message=None
                        ),
                        formData=self.formData,
                    )

        except Exception as error:
            return FireworksSFTJob(
                jobPath=self.jobPath,
                jobStatus="error",
                jobId=self.jobId,
                modelId=self.modelId,
                modelPath=None,
                jobInfo=JobInfo(status="error", info={}, message=str(error)),
                formData=self.formData,
            )


async def get_fine_tuning_job_details(job_path: str) -> FineTuningJobResponse:
    """Get details about a fine-tuning job"""
    url = f"/v1/{job_path}"
    response = await FIREWORKS_CLIENT.get(url)

    if response.status_code != 200:
        raise ValueError(f"Failed to get fine-tuning job details: {response.text}")

    data = response.text

    try:
        return FineTuningJobResponse.model_validate_json(data)
    except Exception as e:
        raise ValueError(f"Invalid API response format: {str(e)}")


async def deploy_model_request(account_id: str, model_id: str) -> Dict[str, Any]:
    """
    Deploy a fine-tuned model

    This is called both to deploy the model and to poll for the deployment status.
    If the model has already been requested to be deployed,
    the API returns 400 along with the deployment status in the message.
    """
    url = f"/v1/accounts/{account_id}/deployedModels"

    model_path = f"accounts/{account_id}/models/{model_id}"
    body = {
        "model": model_path,
        "displayName": model_path,
        "default": True,
        "serverless": True,
        "public": False,
    }

    headers = {
        "Content-Type": "application/json",
    }

    res = await FIREWORKS_CLIENT.post(url, headers=headers, json=body)
    data = res.json()
    if not data:
        raise ValueError("Empty response received from deploy model request")
    return data


def get_deployment_status(deploy_model_response: Dict[str, Any]) -> str:
    """Extract the deployment status from the deploy model response"""
    message = deploy_model_response.get("message")
    if not message or not isinstance(message, str):
        raise ValueError("Failed to get deployment status message")

    status = message.split(":")[-1].strip()
    if not status:
        raise ValueError("Failed to parse deployment status from message")

    return status


async def start_sft_fireworks(
    model_name: str,
    inferences: List[Sample],
    validation_split_percent: float,
    template_env: Environment,
    request: FineTuningRequest,
) -> Dict[str, Any]:
    """Start a supervised fine-tuning job with Fireworks"""
    fireworks_examples = [
        tensorzero_inference_to_fireworks_messages(inference, template_env)
        for inference in inferences
    ]

    dataset_id = await create_dataset_record(
        FIREWORKS_ACCOUNT_ID, len(fireworks_examples)
    )

    await upload_dataset(FIREWORKS_ACCOUNT_ID, dataset_id, fireworks_examples)

    # Poll until dataset is ready
    while not await dataset_is_ready(FIREWORKS_ACCOUNT_ID, dataset_id):
        logger.info(f"Dataset {dataset_id} is not ready yet, waiting...")
        await asyncio.sleep(1)

    job_info = await create_fine_tuning_job(
        FIREWORKS_ACCOUNT_ID, dataset_id, model_name, validation_split_percent
    )

    return job_info


def tensorzero_inference_to_fireworks_messages(
    sample: Sample, env: Environment
) -> FireworksExample:
    """Convert a TensorZero inference to Fireworks messages format"""
    messages = []
    system = try_template_system(sample, env)
    if system:
        messages.append(system)

    # Handle input messages
    for message in sample["input"].get("messages", []):
        for content in message["content"]:
            if content["type"] == "text":
                messages.append(
                    {
                        "role": message["role"],
                        "content": render_message(content, message["role"], env),
                    }
                )
            else:
                raise ValueError(
                    "Only text messages are supported for Fireworks fine-tuning"
                )

    # Handle output
    is_chat_inference = isinstance(sample["output"], list)
    if is_chat_inference:
        output = sample["output"]
        if len(output) != 1:
            raise ValueError("Chat inference must have exactly one message")
        if output[0]["type"] != "text":
            raise ValueError("Chat inference must have a text message as output")
        messages.append({"role": "assistant", "content": output[0]["text"]})
    elif "raw" in sample["output"]:
        output = sample["output"]
        messages.append({"role": "assistant", "content": output["raw"]})
    else:
        raise ValueError("Invalid inference type")

    return {"messages": messages}


async def create_dataset_record(account_id: str, example_count: int) -> str:
    """
    Create a dataset record in Fireworks.
    This is a placeholder for the dataset that gets uploaded in a subsequent call.
    """
    # IMPORTANT: Fireworks requires that IDs:
    #   - Start with a letter
    #   - Contain only letters, numbers, and hyphens
    dataset_id = f"t0-{uuid7()}"

    url = f"/v1/accounts/{account_id}/datasets"

    headers = {
        "Content-Type": "application/json",
    }

    body = {
        "datasetId": dataset_id,
        "dataset": {
            "displayName": dataset_id,
            "exampleCount": str(example_count),
            "userUploaded": {},  # Can use this for function_name, timestamp, etc. later
            "format": "CHAT",  # Options are CHAT, COMPLETION, and FORMAT_UNSPECIFIED
        },
    }

    response = await FIREWORKS_CLIENT.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise ValueError(f"Failed to create dataset record: {response.text}")

    # TODO: we might want to do something with the response?

    return dataset_id


async def upload_dataset(
    account_id: str, dataset_id: str, examples: List[FireworksExample]
) -> Dict[str, Any]:
    """Upload dataset to Fireworks for fine-tuning"""
    url = f"/v1/accounts/{account_id}/datasets/{dataset_id}:upload"

    # Convert examples to JSONL format
    jsonl_data = "\n".join([json.dumps(example) for example in examples])

    files = {"file": ("dataset.jsonl", jsonl_data, "application/jsonl")}

    response = await FIREWORKS_CLIENT.post(url, files=files)

    if response.status_code != 200:
        raise ValueError(f"Failed to upload dataset: {response.text}")

    data = response.json()

    if not data:
        raise ValueError("Empty response received from upload dataset request")

    return data


async def dataset_is_ready(account_id: str, dataset_id: str) -> bool:
    """Check if a dataset is ready for fine-tuning"""
    url = f"/v1/accounts/{account_id}/datasets/{dataset_id}"

    response = await FIREWORKS_CLIENT.get(url)
    response.raise_for_status()
    data = response.json()
    if "state" not in data:
        raise ValueError("Dataset status response missing state field")
    return data["state"] == "READY"


async def create_fine_tuning_job(
    account_id: str, dataset_id: str, base_model: str, val_split: float
) -> Dict[str, Any]:
    """
    Create a fine-tuning job

    Returns a path like "accounts/viraj-ebfe5a/fineTuningJobs/2aecc5ff56364010a143b6b0b0568b5a"
    which is needed for getting the job status
    """
    url = f"/v1/accounts/{account_id}/fineTuningJobs"

    headers = {
        "Content-Type": "application/json",
    }

    body = {
        "dataset": f"accounts/{account_id}/datasets/{dataset_id}",
        "baseModel": base_model,
        "conversation": {},  # empty due to us using the default conversation template
        "evaluationSplit": val_split / 100,
    }

    response = await FIREWORKS_CLIENT.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()
    if "name" not in data:
        raise ValueError("Fine tuning job response missing name field")
    return data
