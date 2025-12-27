1. Set the environment variable `OPENAI_API_KEY` with your OpenAI API key. (The relay gateway will use it.)

2. Set up Postgres:

```bash
docker-compose run --rm relay-gateway --run-postgres-migrations
```

3. Create a TensorZero API key:

```bash
docker-compose run --rm relay-gateway --create-api-key
```

4. Set the environment variable `TENSORZERO_RELAY_API_KEY` with the key you just created. (The edge gateway will use it.)

5. Spin up both gateways and Postgres:

```bash
docker compose up
```
