#!/bin/bash
# entrypoint.sh
#
# Usage example:
#   docker run \
#     -p 8080:80 \
#     -e BEARER_TOKEN=SUPER_SECRET_TOKEN \
#     my-tgi-image \
#     --model-id microsoft/Phi-3.5-mini-instruct \
#     --port 8081 \
#     --max-input-length 1024 \
#     --max-total-tokens 2048 \
#     --max-batch-prefill-tokens 1024 \
#     --quantize fp8

# Replace placeholder in nginx config with real token from env
if [ -z "$BEARER_TOKEN" ]; then
  echo "Missing BEARER_TOKEN env var. Exiting."
  exit 1
fi

sed -i "s#_MY_SECRET_#${BEARER_TOKEN}#g" /etc/nginx/conf.d/default.conf

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'
# Run TGI in background; pass all command-line args directly
text-generation-launcher "$@" &

# Start nginx in the foreground
exec nginx -g 'daemon off;'
