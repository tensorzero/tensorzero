# Dockerfile

FROM ghcr.io/huggingface/text-generation-inference:latest

# Install nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Copy the nginx config (we'll define it below)
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy our entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Nginx listens on port 80; TGI will listen on 8081 (or whatever you pass in)
EXPOSE 80

# Entrypoint runs TGI and nginx
ENTRYPOINT ["/entrypoint.sh"]
