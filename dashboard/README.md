# TensorZero Dashboard

## Status

This dashboard is currently a work in progress.

Our goals for this project are to:

- [ ] Allow users to run TensorZero recipes through the dashboard. To start, this will include:

  - [ ] Supervised fine-tuning
  - [ ] Dynamic in-context learning

- [ ] Allow users to review inferences and episodes and provide feedback to either.
- [ ] Provide an easy dashboard showing the relative performance of different variants for a particular function.
- [ ] Allow users to edit the configuration through the dashboard.

Currently, we are building out the dashboard incrementally.
We have the beginning of a fine-tuning form that uses the OpenAI fine-tuning API.

## Running the Dashboard

### Prerequisites

- Node.js (we have only tested with v22.9.0)
- Docker Compose
- a Rust toolchain

### Setup

Currently, the dashboard only runs against hardcoded fixtures in `fixtures/`.
It depends on a running ClickHouse instance that has been initialized with the TensorZero data model.
We include some fixture data as well in order to exercise some functionality.
You will need Docker Compose installed to run the dependencies.

It also requires a one-time build of a WebAssembly module from Rust source code that is used to ensure consistent templating of messages across the gateway and dashboard.

Here are the steps in order to run or test the dashboard assuming you have the prerequisites installed and this repository checked out:

1. Install npm dependencies: `npm install --legacy-peer-deps cmdk`
2. Build the WebAssembly module following instructions in `app/utils/minijinja/README.md`.
3. Create a `.env` file in the `dashboard` directory and set the following environment variables for the server:

```bash
OPENAI_API_KEY=<your-key>
FIREWORKS_API_KEY=<your-key>
FIREWORKS_ACCOUNT_ID=<your-account-id>
CLICKHOUSE_URL=<your-clickhouse-url> # For testing, set to http://localhost:8123/tensorzero
```

4. Run the dependencies: `docker compose -f fixtures/docker-compose.yml up`

With the dependencies running, you can run the tests with `npm run test`.
Similarly, you can start a development server with `npm run dev`.

We do not currently have a production build process or any way to change configuration.
