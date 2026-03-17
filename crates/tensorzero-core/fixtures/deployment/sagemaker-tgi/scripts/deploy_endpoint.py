# /// script
# dependencies = [
#   "sagemaker",
# ]
# ///
import os
import sys

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::637423354485:role/service-role/AmazonSageMaker-ExecutionRole-20250328T164731"

HF_TOKEN = os.environ["HF_TOKEN"]

# Hub Model configuration. https://huggingface.co/models
hub = {
    #'SM_NUM_GPUS': json.dumps(1),
    # "LOG_LEVEL": "trace",
    # "QUANTIZE": "awq",
    "HF_TOKEN": HF_TOKEN,
    # TGI has a 'Warming up model' step, which cannot be disabled.
    # It creates a maximum-size request to test memory usage, which can easily exceed
    # the 180 second startup timeout for Sagemaker serverless.
    # By lowering the maximum token limits, we can speed up the startup time.
    # All of our e2e tests use relatively small requests, so this is fine.
    "MAX_CLIENT_BATCH_SIZE": "1",
    "MAX_BATCH_PREFILL_TOKENS": "100",
    "MAX_BATCH_TOTAL_TOKENS": "1000",
    "MAX_TOTAL_TOKENS": "1000",
    "MAX_BATCH_SIZE": "1",
}

image_uri = sys.argv[1]

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role,
)

SERVERLESS = True

if SERVERLESS:
    predictor = huggingface_model.deploy(
        serverless_inference_config=ServerlessInferenceConfig(
            memory_size_in_mb=6144,
            max_concurrency=1,
        ),
    )
else:
    predictor = huggingface_model.deploy(initial_instance_count=1, instance_type="ml.t2.medium")
# send request
predictor.predict(
    {
        "inputs": "Hi, what can you help me with?",
    }
)
