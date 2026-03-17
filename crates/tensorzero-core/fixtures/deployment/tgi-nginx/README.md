# TGI NGINX Docker Image

This image is used to run [TGI](https://huggingface.co/docs/text-generation-inference/index) behind an NGINX proxy that provides Bearer token authentication.
We use it to provide a persistent secure endpoint serving TGI for our E2E tests.

## Usage example:

```bash
docker run \
 -p 8080:80 \ # Map port 8080 of the host to port 80 of the container
              # (can change this, but the container will listen on port 80)
 -e BEARER_TOKEN=SUPER_SECRET_TOKEN \ # Set the BEARER_TOKEN environment variable to your secret token
 tensorzero/tgi-nginx:latest \
    --model-id microsoft/Phi-3.5-mini-instruct \ # The model to serve
    --max-input-length 1024 \
    --max-total-tokens 2048 \
    --max-batch-prefill-tokens 1024 \
    --quantize fp8
    # Do not pass --port here, it is set by the container
```
