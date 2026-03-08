FROM lmsysorg/sglang:latest

# Install nginx and remove default site
RUN apt-get update && apt-get install -y nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm /etc/nginx/sites-enabled/default

# Copy the nginx config
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy our entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Nginx listens on port 80; SGLang will listen on 8081
EXPOSE 80

# Entrypoint runs SGLang and nginx
ENTRYPOINT ["/entrypoint.sh"]
