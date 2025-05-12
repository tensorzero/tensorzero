import logging
import os
import typing as t

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorzero import AsyncTensorZeroGateway

from optimization_server.sft.common import FineTuningRequest
from optimization_server.sft.fireworks_sft import FireworksSFTJob
from optimization_server.sft.openai_sft import BaseSFTJob, OpenAISFTJob

logging.basicConfig(
    format="%(asctime)s.%(msecs)03dZ  %(levelname)-5s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

CONFIG_PATH = os.environ.get("TENSORZERO_UI_CONFIG_PATH", "config/tensorzero.toml")
CLICKHOUSE_URL = os.environ["TENSORZERO_CLICKHOUSE_URL"]

_tensorzero_client = AsyncTensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH, clickhouse_url=CLICKHOUSE_URL, async_setup=False
)
assert isinstance(_tensorzero_client, AsyncTensorZeroGateway)
TENSORZERO_CLIENT = _tensorzero_client

app = FastAPI()


class OptimizationRequest(BaseModel):
    # Turn this into 'Union[FineTuningRequest, OtherRequest]' when we have more optimizations implemented
    data: FineTuningRequest = Field(discriminator="kind")


JOB_STORE: t.Dict[str, BaseSFTJob] = {}


async def start_sft_job(data: FineTuningRequest) -> BaseSFTJob:
    if data.model.provider == "openai":
        job = await OpenAISFTJob.from_form_data(TENSORZERO_CLIENT, data)
    elif data.model.provider == "fireworks":
        job = await FireworksSFTJob.from_form_data(TENSORZERO_CLIENT, data)
    else:
        raise RuntimeError("Unsupported model provider: %s" % data.model.provider)

    JOB_STORE[data.jobId] = job
    return job


@app.get("/optimizations/poll/{job_id}")
async def poll_optimization(job_id: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    JOB_STORE[job_id] = await job.poll()
    return job.status()


@app.post("/optimizations/")
async def start_optimization(request: OptimizationRequest) -> BaseSFTJob:
    if request.data.kind == "sft":
        return await start_sft_job(request.data)
    raise ValueError("Unsupported optimization kind: %s" % request.data.kind)
