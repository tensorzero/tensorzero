# Based on https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-7b/LMI/llama2-7b.ipynb

import sagemaker
from sagemaker import image_uris
import boto3

HF_MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
SAGEMAKER_ROLE = "arn:aws:iam::637423354485:role/service-role/AmazonSageMaker-ExecutionRole-20250328T164731"

def main():
    role = SAGEMAKER_ROLE
    sess = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs
    bucket = sess.default_bucket()  # bucket to house artifacts


    model_bucket = sess.default_bucket()  # bucket to house model artifacts
    s3_code_prefix = f"hf-large-model-djl/{HF_MODEL_NAME}/code"  # folder within bucket where code artifact will go
    s3_model_prefix = f"hf-large-model-djl/{HF_MODEL_NAME}/model"  # folder within bucket where model artifact will go
    region = sess._region_name
    account_id = sess.account_id()

    s3_client = boto3.client("s3")
    sm_client = boto3.client("sagemaker")
    smr_client = boto3.client("sagemaker-runtime")

    # deepspeed_image_uri = image_uris.retrieve(
    #     framework="djl-deepspeed", 
    #     region=sess.boto_session.region_name, 
    #     version="0.29.0"
    # )
    deepspeed_image_uri = "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:2.3.0-transformers4.48.0-cpu-py311-ubuntu22.04-v2.0"
    print("Got url: ", deepspeed_image_uri)

    env_generation = {"HUGGINGFACE_HUB_CACHE": "/tmp",
                  "TRANSFORMERS_CACHE": "/tmp",
                  "SERVING_LOAD_MODELS": "test::Python=/opt/ml/model",
                  "HF_MODEL_ID": HF_MODEL_NAME,
                  "OPTION_TRUST_REMOTE_CODE": "true",
                  "OPTION_TENSOR_PARALLEL_DEGREE": "max",
                  "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
                  "OPTION_DTYPE":"fp16",
                  "HF_TASK": "question-answering",
                 }

    # - Select the appropriate environment variable which will tune the deployment server.
    env = env_generation

    # - now we select the appropriate container 
    inference_image_uri = deepspeed_image_uri


    print(f"Environment variables are ---- > {env}")
    print(f"Image going to be used is ---- > {inference_image_uri}")
    model_name = sagemaker.utils.name_from_base("unsloth-llama3-1b-4bit")
    print(model_name)

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": inference_image_uri,
            "Environment": env,
        }
    )
    model_arn = create_model_response["ModelArn"]

    print(f"Created Model: {model_arn}")

    endpoint_config_name = f"{model_name}-config"
    endpoint_name = f"{model_name}-endpoint"

    endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': 3072,
                    'MaxConcurrency': 1,
                },
            },
        ],
    )
    print("Created endpoint config: ", endpoint_config_response)

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
    )
    print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")


if __name__ == "__main__":
    main()
