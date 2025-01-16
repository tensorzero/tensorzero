# TensorZero UI

The TensorZero UI allows you to interact with TensorZero through a web interface.
The goal for this project is to provide a user-friendly way to browse through inferences and episodes, understand the performance of variants, provide feedback on these, and launch optimization jobs based on the dataset TensorZero has accumulated.

Currently, we are building out the UI incrementally.

However, in order to get feedback from the community and provide some value to them, we have released an early version of the UI as a docker image available on Docker Hub as `tensorzero/ui`.

## Running the UI

### Users

The gateway requires a TensorZero configuration tree and the URL of the ClickHouse instance that the TensorZero gateway is running against.
Given these two things, the easiest way to run the UI is to use the `tensorzero/ui` docker image.
We include an example `docker-compose.yml` and `.env.example` file in this directory.
You optionally can include an `OPENAI_API_KEY` and `FIREWORKS_API_KEY` in the `.env` file to enable fine-tuning on curated data from your ClickHouse using those services.

The docker container exposes the UI on port 4000.

### Developers

#### Prerequisites

- Node.js (we have only tested with v22.9.0)
- Docker Compose
- a Rust toolchain

#### Setup

For development, the UI runs against hardcoded fixtures in `fixtures/`.
It depends on a running ClickHouse instance that has been initialized with the TensorZero data model.
We include some fixture data as well in order to exercise some functionality.
You will need Docker Compose installed to run the dependencies.

It also requires a one-time build of a WebAssembly module from Rust source code that is used to ensure consistent templating of messages across the gateway and UI.

Here are the steps in order to run or test the UI assuming you have the prerequisites installed and this repository checked out:

1. Install npm dependencies: `npm install --legacy-peer-deps cmdk`
2. Build the WebAssembly module following instructions in `app/utils/minijinja/README.md`.
3. Create a `.env` file in the `ui` directory and set the following environment variables for the server:

```bash
OPENAI_API_KEY=<your-key>
FIREWORKS_API_KEY=<your-key>
FIREWORKS_ACCOUNT_ID=<your-account-id>
CLICKHOUSE_URL=<your-clickhouse-url> # For testing, set to http://localhost:8123/tensorzero
TENSORZERO_UI_CONFIG_DIR=<path-to-config-dir> # For testing, set to ./fixtures/config
```

4. Run the dependencies: `docker compose -f fixtures/docker-compose.yml up`

With the dependencies running, you can run the tests with `npm run test`.
Similarly, you can start a development server with `npm run dev`.

### Running the production server

To run the production server, you should set the environment variables in the `.env.example` file in `.env` (API keys are optional but required if you'd like to fine-tune) and then run `docker compose up`.

If you are running the production server against the fixtures from the development environment, set `CONFIG_DIR=./fixtures/config` and `CLICKHOUSE_URL=http://host.docker.internal:8123/tensorzero` in the `.env` file.
