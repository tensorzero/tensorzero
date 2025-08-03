# AWS Sagemaker ollama Docker Image

This image is used to run [ollama](https://github.com/ollama/ollama) on AWS Sagemaker, with
gemma3:1b automatically pulled.
We use it to provide a persistent endpoint for our e2e tests

## Deploying (for TensorZero team)

Run `./scripts/deploy.sh VERSION`, which will tag a Docker image with 'VERSION' and update Sagemaker
