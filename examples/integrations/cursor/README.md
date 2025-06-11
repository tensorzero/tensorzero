# Example: Integrating Cursor with TensorZero

> [!NOTE]
>
> Read more about this integration on our blog: [Reverse Engineering Cursor's LLM Client](https://tensorzero.com/blog/reverse-engineering-cursors-llm-client/)

This example shows how to use the TensorZero Gateway as a proxy between Cursor and the LLM APIs &mdash; enabling you to observe LLM calls being made, run evaluations on individual inferences, use inference-time optimizations, and even experiment with and optimize the prompts and models that Cursor uses.

## Setup

1. Create a `.env` file with the credentials for the LLM providers you want to use (as in any other TensorZero deployment).
   Our example uses OpenAI, Anthropic, and Google AI Studio, but you can use any LLM provider TensorZero supports.
   See `.env.example` for an example.
2. Generate a strong API key to protect your gateway endpoint from unauthorized access.
   Set it as the value for `API_TOKEN` in your `.env` file.
   We've had some issues with special characters in this step so recommend you use an alphanumeric string to be safe.
   Optionally, set `USER="yourname"` if you'd like to tag each request with your name for downstream use.
3. Create an [ngrok](https://ngrok.com/) account.
   Grab your auth token and add it to the `.env` file as the value for `NGROK_AUTHTOKEN`.
4. Run `docker compose up` to stand up ClickHouse, the TensorZero Gateway, the TensorZero UI, Nginx, and ngrok.
   To avoid port conflicts, ClickHouse and TensorZero services that would normally bind to ports `XXXX` bind to `1XXXX` instead (e.g. `3000` → `13000`).
5. Visit `http://localhost:4040` and grab your ngrok URL.
6. Set your Cursor `OPENAI_BASE_URL` to your ngrok URL with the `/openai/v1` suffix (e.g. `https://your-id.ngrok-free.app/openai/v1`).
   This should be available in your Cursor model settings.
7. Set your OpenAI API key to the `API_TOKEN` value you set in your `.env` file (not your OpenAI API key!).
8. Turn off all the models in Cursor and add a custom one called `tensorzero::function_name::cursorzero`.
9. Verify the new OpenAI connection by turning on your custom API key.

Now, you can use Cursor as you normally would but with the `tensorzero::function_name::cursorzero` model you defined.
This will send all traffic through your self-hosted TensorZero Gateway, which in turn will route those requests to the LLM APIs you defined in your `tensorzero.toml` file.
Take a look at the server running on `http://localhost:14000` to see what your requests look like!

<details>
<summary>Intercepting Cursor Tab Requests</summary>

> This section is mostly for reference.
> You do not need to do this to run the Cursor integration today.

1. Install `mitmproxy`, and install the root certificate system-wide.
2. Turn off HTTP2 in VS Code settings. [[reference]](https://forum.cursor.com/t/add-authorized-certificates-to-cursor/21765)
3. Start Cursor with the environment variables `http_proxy=http://localhost:8080` and `https_proxy=http://localhost:8080`.

</details>

## A Haiku about This Integration

_Cursor calls go through_

_TensorZero watches them_

_See the requests flow_
