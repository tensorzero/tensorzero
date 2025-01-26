# SGLang NGINX Docker Image

This image is used to run [SGLang](https://huggingface.co/docs/text-generation-inference/index) behind an NGINX proxy that provides Bearer token authentication.
We use it to provide a persistent secure endpoint serving SGLang for our E2E tests.

## Usage example:

```bash
docker run \
 -p 8080:80 \ # Map port 8080 of the host to port 80 of the container
              # (can change this, but the container will listen on port 80)
 -e BEARER_TOKEN=SUPER_SECRET_TOKEN \ # Set the BEARER_TOKEN environment variable to your secret token
 tensorzero/sgl-nginx:latest \
    --model-path HuggingFaceTB/SmolLM-1.7B-Instruct # The model to serve
    --trust-remote-code
    --disable-overlap
    # Do not pass --port here, it is set by the container
```
