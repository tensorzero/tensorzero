#!/bin/sh
set -e

# Replace environment variables in the template (USER is optional)
# Use default values if variables are not set
API_TOKEN="${API_TOKEN:-}"
USER="${USER:-}"

# Create the conf.d directory if it doesn't exist
mkdir -p /etc/nginx/conf.d

# Remove the default config and replace with our custom one
rm -f /etc/nginx/conf.d/default.conf

# Generate our server configuration from template
{
    echo "map \$http_authorization \$is_authorized {"
    echo "    default                     0;"
    echo "    \"~*^Bearer ${API_TOKEN}\$\"   1;"
    echo "}"
    echo ""
    sed -e "s|\${API_TOKEN}|${API_TOKEN}|g" -e "s|\${USER}|${USER}|g" /etc/nginx/nginx.conf.template
} > /etc/nginx/conf.d/tensorzero.conf

# Copy the Lua script to the nginx directory
cp /modify_body.lua /etc/nginx/modify_body.lua

# Execute the default nginx command
exec "$@"
