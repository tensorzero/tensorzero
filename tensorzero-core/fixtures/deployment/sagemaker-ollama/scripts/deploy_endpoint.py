# Based on https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-7b/LMI/llama2-7b.ipynb
# This deploys an existing Docker image (specified by the `DOCKER_IMAGE_URI` env var) to a Sagemaker endpoint
# in the TensorZero AWS account. It's specialized for that account, and isn't a general-purpose script
import os
import time

import boto3
import sagemaker

SAGEMAKER_ROLE = "arn:aws:iam::637423354485:role/service-role/AmazonSageMaker-ExecutionRole-20250328T164731"
SERVERLESS = False

PROVISIONED_INSTANCE_TYPE = "ml.t2.medium"
PROVISIONED_ENDPOINT_CONFIG_NAME = "gemma3-1b-provisioned-config"
PROVISIONED_ENDPOINT_NAME = "gemma3-1b-provisioned-endpoint"


def main():
    role = SAGEMAKER_ROLE
    inference_image_uri = os.environ["DOCKER_IMAGE_URI"]
    sm_client = boto3.client("sagemaker")
    env = {}

    print(f"Container env: {env}")
    print(f"Container image: {inference_image_uri}")
    model_name = sagemaker.utils.name_from_base("gemma-ollama")
    print("Sagemaker model name: ", model_name)

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": inference_image_uri,
            "Environment": env,
        },
    )
    model_arn = create_model_response["ModelArn"]

    print(f"Created Model: {model_arn}")

    if SERVERLESS:
        endpoint_config_name = f"{model_name}-config"
        endpoint_name = f"{model_name}-endpoint"
        endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "variant1",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": 3072,
                        "MaxConcurrency": 1,
                    },
                },
            ],
        )
    else:
        try:
            sm_client.delete_endpoint(EndpointName=PROVISIONED_ENDPOINT_NAME)
        except Exception as e:
            if "Could not find endpoint" in str(e):
                print("No existing endpoint to delete")
            else:
                raise

        try:
            sm_client.delete_endpoint_config(EndpointConfigName=PROVISIONED_ENDPOINT_CONFIG_NAME)
        except Exception as e:
            if "Could not find endpoint configuration" in str(e):
                print("No existing endpoint config to delete")
            else:
                raise

        endpoint_config_name = PROVISIONED_ENDPOINT_CONFIG_NAME
        endpoint_name = PROVISIONED_ENDPOINT_NAME
        endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "variant1",
                    "ModelName": model_name,
                    "InstanceType": PROVISIONED_INSTANCE_TYPE,
                    "InitialInstanceCount": 1,
                },
            ],
        )

    print("Created endpoint config: ", endpoint_config_response)
    for i in range(5):
        try:
            create_endpoint_response = sm_client.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            break
        except Exception as e:
            if "Cannot create already existing endpoint" in str(e):
                print("Endpoint not yet deleted, retrying...")
                time.sleep(1)
            else:
                raise
    print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")


if __name__ == "__main__":
    main()
