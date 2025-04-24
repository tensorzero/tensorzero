# TensorZero as a Cursor proxy layer

Note: This is active work in progress.

This example demonstrates how TensorZero can be used as a proxy layer for Cursor. To try it out:

- put credentials for whatever LLM providers you want in `.env` as in any other TensorZero deployment. Our example uses OpenAI, Anthropic, and Google AI studio. But, you can use any LLM provider TensorZero supports.
- Set an API key to protect your gateway endpoint. For now, this requires editing `nginx/nginx.conf` where it says `your-token-here` but we'll likely switch this for an environment variable soon.
- Run `docker compose up`. This stands up ClickHouse, the TensorZero Gateway, the TensorZero UI, and nginx.
- Install and run `ngrok`; you may need to install it with a package manager. The nginx front end runs on port 6900 so you should run `ngrok http: http://localhost:6900`. Keep track of the URL where your service is running.
- Set your Cursor `OPENAI_BASE_URL` to `https://your-id.ngrok-free.app/openai/v1`. This should be available in your Cursor model settings.
- Set your OpenAI API key to the value you set in your `nginx.conf`.
- Turn off all the models in Cursor and add a custom one called `tensorzero::function_name::cursorzero`.
- Verify the new OpenAI connection by turning on your custom API key.

Now, you can use Cursor as you would normally but with the `tensorzero::function_name::cursorzero` model you installed.
This will send all traffic through your local TensorZero gateway.
Take a look at the server running on `http://localhost:6901` to see what your requests look like!

## A haiku about this integration

Cursor calls go through
TensorZero watches them
See the requests flow
