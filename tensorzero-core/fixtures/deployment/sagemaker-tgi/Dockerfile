FROM ghcr.io/huggingface/text-generation-inference:3.3.0-intel-cpu

RUN mkdir -p /app
ADD ./entrypoint.sh /app/
ADD ./proxy.py /app/
ADD ./pyproject.toml /app/
ADD ./uv.lock /app/

RUN cd /app && uv sync --frozen --no-dev
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGINGFACE_HUB_CACHE="/hf-cache"
ENV HF_HOME="/hf-home"
ENV MODEL_ID="google/gemma-3-1b-it"
RUN mkdir -p /hf-home
RUN mkdir -p /hf-cache
RUN chmod -R 777 /hf-cache
RUN chmod -R 777 /hf-home
#RUN text-generation-server download-weights $MODEL_ID
ENTRYPOINT [ "/app/entrypoint.sh" ]
