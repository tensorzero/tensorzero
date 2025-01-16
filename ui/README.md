# TensorZero UI

The TensorZero UI allows you to interact with TensorZero through a web interface.
The goal for this project is to provide a user-friendly way to browse through inferences and episodes, understand the performance of variants, provide feedback on these, and launch optimization jobs based on the dataset TensorZero has accumulated.

We are building out the UI incrementally.

However, in order to get feedback from the community and provide some value to them, we have released an early version of the UI as a docker image available on Docker Hub as `tensorzero/ui`.

## Running the UI

The UI requires a TensorZero configuration tree and the URL of the ClickHouse instance that the TensorZero gateway is running against.
Given these two things, the easiest way to run the UI is to use the `tensorzero/ui` docker image.
We include an example `docker-compose.yml` and `.env.example` file in this directory.
You optionally can include an `OPENAI_API_KEY` and `FIREWORKS_API_KEY` in the `.env` file to enable fine-tuning on curated data from your ClickHouse using those services.
The server looks for the `TENSORZERO_UI_CONFIG_PATH` environment variable to find the configuration file (defaults to `config/tensorzero.toml`).
Many of the examples in the `examples/` directory of the top-level repository include a `docker-compose.yml` file that concurrently runs Clickhouse, the UI and the TensorZero gateway.
These might also be useful to show how they work together.

The docker container exposes the UI on port 4000.
