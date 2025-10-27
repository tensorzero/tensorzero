# Contributing to TensorZero

Thank you for your interest in contributing to TensorZero!

TensorZero aims to power the next generation of AI applications.
We'd love to collaborate with you to make this vision a reality.

> [!TIP]
>
> In addition to community contributions, we're also hiring in NYC (in-person only).
> See our [open roles](https://www.tensorzero.com/jobs).

## License

TensorZero is licensed under the [Apache 2.0 license](LICENSE).
By contributing to this repository, you agree to license your contributions under the same license.

## Community & Support

### Slack and Discord

Join our community on [Slack](https://www.tensorzero.com/slack) or [Discord](https://www.tensorzero.com/discord) to chat with the team and other contributors.

### GitHub

We use GitHub Issues to track bugs and feature requests.
For general questions, technical support, and conversations not directly related to code, please use GitHub Discussions.

## Contributions

> [!TIP]
>
> See the [`good-first-issue`](https://github.com/tensorzero/tensorzero/issues?q=is%3Aopen+is%3Aissue+label%3Agood-first-issue) label for simpler issues that might be a good starting point for new contributors.

### Code

For small changes (i.e. a few lines of code), feel free to open a PR directly.

For larger changes, please communicate with us first to avoid duplicate work or wasted effort.
You can start a discussion (GitHub, Slack, or Discord) or open an issue as a starting point.
The team will be happy to provide feedback and guidance.

At this time, we don't assign issues to new external contributors (in the past, most people we assigned issues to never submitted a PR).
Please submit a PR directly once you're ready to start working on an issue.

> [!TIP]
>
> See the "Technical Guide" section below for more details on building and testing TensorZero.

### Documentation

The content for our documentation lives in the `docs/` directory.

For small changes (e.g. typos), feel free to open a PR directly.

For larger changes, please communicate with us first to avoid duplicate work or wasted effort.
You can start a discussion (GitHub, Slack, or Discord) or open an issue as a starting point.

### Content — Examples, Tutorials, etc.

We'd love to collaborate on examples, tutorials, and other content that showcases how to build AI applications with TensorZero.

For content contributed directly to our repository, please follow the same process as code contributions.

For external content (e.g. blog posts, videos, social media content), we're excited to support and amplify your work.
Share your content in our community channels (Slack and Discord), tag us on social media, or reach out if you'd like technical review or feedback before publishing.

We're happy to provide guidance and support for both types of content to help you create high-quality resources for the TensorZero community.

### Integrations

We're open to exploring integrations with other projects and tools (both open-source and commercial).
Reach out if you're interested in collaborating.

### Security

If you discover a security vulnerability, please email us at [security@tensorzero.com](mailto:security@tensorzero.com).

### Other

Did you have something else in mind? Reach out on Slack or Discord and let us know.

---

## Technical Guide

### Setup

- Install Rust (1.80+) [→](https://www.rust-lang.org/tools/install)
- Install `cargo-deny` [→](https://github.com/EmbarkStudios/cargo-deny)
- Install `cargo-nextest` [→](https://nexte.st/docs/installation/pre-built-binaries/)
- Install `pre-commit` [→](https://pre-commit.com/#install)
- Enable `pre-commit` in your repository: `pre-commit install`
- Install Docker [→](https://docs.docker.com/get-docker/)
- Install `uv` [→](https://docs.astral.sh/uv/)
- Install Python (3.9+) (e.g. `uv python install 3.9` + )
- Install Node.js (we use v22) and `npm` [→](https://nodejs.org/en)
- Install pnpm `npm install -g pnpm@10` [→](https://pnpm.io/installation)

**macOS users:** If you see Rust build errors about missing dynamic libraries for Python, set up a Python virtual environment at `tensorzero/.venv` (e.g. `uv venv` from the `tensorzero` directory)
This ensures the correct Python libraries are available for the build.

### Optimization Recipes

We maintain optimization recipes as Jupyter notebooks in `recipes/`.
These notebooks serve as manual workflows for optimizing (e.g. fine-tuning) TensorZero functions.

Jupyter notebooks are notoriously hard to test, maintain, and review.
To address these issues, each notebook has an accompanying Python script ending in `_nb.py` that serves the same purpose.
We automatically keep these two files in sync using [Jupytext](https://jupytext.readthedocs.io/en/latest/).

To convert a notebook to a script, run `ci/compile-notebook-to-script.sh path/to/notebook.ipynb`.
To convert a script to a notebook, run `ci/compile-script-to-notebook.sh path/to/script_nb.py`.

In `pre-commit` and CI, we check that the notebooks match the relevant scripts using a script `ci/compile-check-notebooks.sh`.

### Tests

#### Rust

##### Unit Tests

```bash
cargo test-unit
```

##### E2E Tests

1. Launch the test ClickHouse database

   ```bash
   docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --wait
   ```

2. Set the relevant environment variables. See `examples/production-deployment/.env.example` for the full list.

3. Launch the gateway in testing mode

   ```bash
   cargo run-e2e
   ```

4. Run the E2E tests
   ```bash
   cargo test-e2e
   ```

> [!TIP]
>
> The E2E tests involve every supported model provider, so you need every possible credential to run the entire test suite.
>
> If your changes don't affect every provider, you can run a subset of tests with `cargo test-e2e xyz`, which will only run tests with `xyz` in their name.

#### Python

1. Launch ClickHouse and the gateway in E2E testing mode (see above).

2. Go to the relevant directory (e.g. `cd clients/python`)

3. Create a virtual environment and install the dependencies

   ```bash
   uv venv
   uv pip sync requirements.txt
   ```

4. Run the tests

   ```bash
   uv run pytest
   ```

5. Run the type checker

   ```bash
   uv pip install pyright
   uv run pyright
   ```

6. Run the formatter

   ```bash
   uv pip install ruff
   uv run ruff format --check .
   uv run ruff check --output-format=github --extend-select I .
   ```

#### TensorZero UI

The UI depends on ClickHouse and other TensorZero components.
For development, we recommend running the TensorZero Gateway and ClickHouse as containers.
We also provide fixtures in `ui/fixtures/`.

To set it up, follow these steps from the `ui` directory:

1. Install dependencies: `pnpm install`
2. Build the internal N-API client for TensorZero using `pnpm -r build`. If you have changed your Rust code, you may also have to run `pnpm build-bindings` from `../internal/tensorzero-node`.
3. Create a `fixtures/.env` following the `fixtures/.env.example`.
4. Set the following environment variables in your cwd `ui/` (note the previous steps edited the vars in `fixtures/`):

   ```bash
   TENSORZERO_GATEWAY_URL="http://localhost:3000"
   TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures"
   TENSORZERO_UI_CONFIG_PATH="fixtures/config/tensorzero.toml"

   # Optional: add provider credentials for optimization workflows
   OPENAI_API_KEY="..."
   FIREWORKS_API_KEY="..."
   FIREWORKS_ACCOUNT_ID="..."
   ```

5. Launch the dependencies: `docker compose -f fixtures/docker-compose.yml up --build --force-recreate`.
   You can omit these last 2 flags to skip the build step, but they ensure you're using the latest gateway.
6. Launch the development server: `pnpm dev`

Separately, you can run headless tests with `pnpm test` and Playwright tests with `pnpm test-e2e` (the latter will require a `pnpm exec playwright install`).
We also maintain a Docker Compose for e2e tests `fixtures/docker-compose.e2e.yml` that is used in CI for the Playwright tests.
This file uses a different configuration that mandates credentials for image fetching.

### Advanced

- If your code affects the serialization of stored data, batch tests might fail because they'll rely on an older serialization of the request. In such cases, you might need to clear the database and re-run the tests. The TensorZero Team can clean up the cache by running `TRUNCATE TABLE tensorzero_e2e_tests.BatchModelInference; TRUNCATE TABLE tensorzero_e2e_tests.BatchRequest;` in the ClickHouse Cloud cluster `dev-tensorzero-e2e-tests`.

---

Thanks again for your interest in contributing to TensorZero! We're excited to see what you build.
