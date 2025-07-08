# /// script
# dependencies = [
#   "sagemaker",
# ]
# ///
import os

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::637423354485:role/service-role/AmazonSageMaker-ExecutionRole-20250328T164731"

HF_TOKEN = os.environ["HF_TOKEN"]

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": "google/gemma-3-1b-it",
    #'SM_NUM_GPUS': json.dumps(1),
    "HF_TOKEN": HF_TOKEN,
    "PORT": "8080",
}

image_uri = "637423354485.dkr.ecr.us-east-2.amazonaws.com/huggingface/text-generation-inference:3.3.0-intel-cpu"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=3072,
        max_concurrency=1,
    ),
)

# send request
predictor.predict(
    {
        "inputs": "Hi, what can you help me with?",
    }
)
