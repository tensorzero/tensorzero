#!/bin/bash

# Replace placeholder in nginx config with real token from env
if [ -z "$BEARER_TOKEN" ]; then
  echo "Missing BEARER_TOKEN env var. Exiting."
  exit 1
fi

# Replace placeholder in nginx config with real token from env
sed -i "s#_MY_SECRET_#${BEARER_TOKEN}#g" /etc/nginx/conf.d/default.conf

ldconfig 2>/dev/null || echo 'Note: Unable to refresh dynamic linker cache. This is expected in some container environments and will not affect functionality.'
# Run SGL in background; pass all command-line args directly
python3 -m sglang.launch_server --host 0.0.0.0 --port 8081 $@ &

# Start nginx in the foreground
exec nginx -g 'daemon off;'
