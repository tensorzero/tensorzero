#!/bin/sh
set -e

# Replace environment variables in the template
envsubst '${API_TOKEN}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Execute the default nginx command
exec "$@"
