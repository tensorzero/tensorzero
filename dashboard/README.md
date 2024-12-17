# TensorZero Dashboard [WIP]

## Status

This dashboard is currently a work in progress.

The goal of the dashboard is to improve the developer experience for TensorZero.

In particular, we want to make it easier to:

- [ ] Run TensorZero recipes through the dashboard.
  - [ ] Supervised fine-tuning
  - [ ] Dynamic in-context learning
- [ ] Review inferences and episodes (incl. providing feedback)
- [ ] Track the performance of different variants for a particular function
- [ ] Edit the configuration through the dashboard

## Running the Dashboard

### Prerequisites

- Node.js (we have only tested with v22.9.0)
- Docker Compose
- Rust toolchain

### Setup

Currently, the dashboard only runs against hardcoded fixtures in `fixtures/`.
It depends on a running ClickHouse instance that has been initialized with the TensorZero data model.
We include some fixture data as well to exercise some functionality.

It also requires a one-time build of a WebAssembly module from Rust source code that is used to ensure consistent templating of messages across the gateway and dashboard.

Here are the steps in order to run or test the dashboard assuming you have the prerequisites installed and this repository checked out:

1. Install npm dependencies: `npm install`
2. Build the WebAssembly module following instructions in `app/utils/minijinja/README.md`.
3. Create a `.env` file in the `dashboard` directory containing `OPENAI_API_KEY=<your-key>` (for the gateway). Also make sure your `OPENAI_API_KEY` environment variable is set (for the server).
4. Run the dependencies: `docker compose -f fixtures/docker-compose.yml up`

With the dependencies running, you can run the tests with `npm run test`.
Similarly, you can start a development server with `npm run dev`.

We do not currently have a production build process or any way to change configuration.
