# TensorZero as a Cursor proxy layer

This example demonstrates how TensorZero can be used as a proxy layer for Cursor. To try it out:

- Put credentials for whatever LLM providers you want in `.env` as in any other TensorZero deployment. Our example uses OpenAI, Anthropic, and Google AI studio. But, you can use any LLM provider TensorZero supports. See `.env.example` for an example.
- Generate a strong API key to protect your gateway endpoint in your .env file and use it replace the `API_TOKEN` value.
- Make an [ngrok](https://ngrok.com/) account. Grab your auth token and add it to the .env file as the value for `NGROK_AUTHTOKEN`.
- Run `docker compose up`. This stands up ClickHouse, the TensorZero Gateway, the TensorZero UI, nginx, and ngrok.
- Visit [http://localhost:4040](http://localhost:4040) and grab your ngrok URL.
- Set your Cursor `OPENAI_BASE_URL` to `https://your-id.ngrok-free.app/openai/v1`. This should be available in your Cursor model settings.
- Set your OpenAI API key to the `API_TOKEN` value you set in your `nginx.conf`.
- Turn off all the models in Cursor and add a custom one called `tensorzero::function_name::cursorzero`.
- Verify the new OpenAI connection by turning on your custom API key.

Now, you can use Cursor as you would normally but with the `tensorzero::function_name::cursorzero` model you installed.
This will send all traffic through your local TensorZero gateway.
Take a look at the server running on `http://localhost:6901` to see what your requests look like!

## Intercepting requests for cursor tab:
> [!NOTE]
> This section is mostly for reference. You do not need to do this to run the Cursor integration today.

- Install mitmproxy, and install the root cert system-wide
- Turn off HTTP2 in vscode settings (https://forum.cursor.com/t/add-authorized-certificates-to-cursor/21765)
- Start cursor with the env vars 'http_proxy=http://localhost:8080' and 'https_proxy=http://localhost:8080'

## A haiku about this integration

Cursor calls go through
TensorZero watches them
See the requests flow
