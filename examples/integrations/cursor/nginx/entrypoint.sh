#!/bin/sh
set -e

# Replace environment variables in the template (USER is optional)
envsubst '${API_TOKEN} ${USER}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Copy the Lua script to the nginx directory
cp /modify_body.lua /etc/nginx/modify_body.lua

# Execute the default nginx command
exec "$@"
