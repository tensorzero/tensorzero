# Contributing to TensorZero

Thank you for your interest in contributing to TensorZero!

TensorZero aims to power the next generation of AI applications. We'd love to collaborate with you to make this vision a reality.

> [!TIP]
>
> In addition to community contributions, we're also hiring in NYC (in-person only). See our [open roles](https://www.tensorzero.com/jobs).

## License

TensorZero is licensed under the [Apache 2.0 license](LICENSE).
By contributing to this repository, you agree to license your contributions under the same license.

## Community & Support

### Slack and Discord

Join our community on [Slack](https://www.tensorzero.com/slack) or [Discord](https://www.tensorzero.com/discord) to chat with the team and other contributors.

### GitHub

We use GitHub Issues to track bugs and feature requests. For general questions, technical support, and conversations not directly related to code, please use GitHub Discussions.

## Contributions

> [!TIP]
>
> See the [`good-first-issue`](https://github.com/tensorzero/tensorzero/issues?q=is%3Aopen+is%3Aissue+label%3Agood-first-issue) label for simpler issues that might be a good starting point for new contributors.

### Code

For small changes (i.e. a few lines of code), feel free to open a PR directly.

For larger changes, please communicate with us first to avoid duplicate work or wasted effort.
You can start a discussion (GitHub, Slack, or Discord) or open an issue as a starting point.
The team will be happy to provide feedback and guidance.

> [!TIP]
>
> See the "Contributor Guide" section below for more details on building and testing TensorZero.

### Documentation

We're planning to open-source our documentation pages soon. See [issue #432](https://github.com/tensorzero/tensorzero/issues/432) for more details.

In the meantime, please open an issue if you have suggestions or find any problems with the documentation.

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

## Contributor Guide

### Setup

- Install Rust (1.80+) [→](https://www.rust-lang.org/tools/install)
- Install `cargo-deny` [→](https://github.com/EmbarkStudios/cargo-deny)
- Install `cargo-nextest` [→](https://nexte.st/docs/installation/pre-built-binaries/)
- Install `pre-commit` [→](https://pre-commit.com/#install)
- Enable `pre-commit` in your repository: `pre-commit install`
- Install Docker [→](https://docs.docker.com/get-docker/)
- Install `uv` [→](https://docs.astral.sh/uv/)
- Install Python (3.10+) (e.g. `uv python install 3.10` + )

### Tests

#### Rust

##### Unit Tests

```bash
cargo test-unit
```

##### E2E Tests

1. Launch the test ClickHouse database

   ```bash
   docker compose -f gateway/tests/e2e/docker-compose.yml up --wait
   ```

2. Set the relevant environment variables. See `examples/production-deployment/.env.example` for the full list.

> [!TIP]
>
> The gateway requires credentials for every model provider defined in its configuration.
> The E2E tests define every supported provider, so you need every possible credential to run the entire test suite.
>
> You can run a subset of the tests by setting the missing credentials for fake values.
> For example, you can set `OPENAI_API_KEY="none"` if you don't plan to run OpenAI tests.
>
> For GCP Vertex AI, you'll need a credentials file. You can use the following fake file and point `GCP_VERTEX_CREDENTIALS_PATH` to it:
>
> ```json
> {
>   "type": "service_account",
>   "project_id": "none",
>   "private_key_id": "none",
>   "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIICXAIBAAKBgQDAKxbF0dfne7PmPwpFEcSi2JFBeO98DXW7bimAPE6dHHCkDvoU\nlD/fy8svrPU6xsCYxM3LfKY/F+s/P+FizXUQ6eDu5ipYCRfweiQ4gqms+zROeORA\nJez3zelPQ7vY/MYCnp0LYYCH2HTyBeMFIX+Rgwjral495j0O6uV7cjgneQIDAQAB\nAoGAOXcpMjLUS6bUX1AOtCTiFoiIt3mAtCoaQNhqlKx0Hct5a7YG1syWZUg+FJ22\nH8N7qLOBjw5RcKCoepuRvMgP71+Hp03Xt8WSpN1Evl6EllwtmTtVTTeVS8fjP7xL\nhc7XemtDPY/81cBuj+HCit9/+44HZCT9V3dV6D9IWWnc3mECQQD1sTvcNAsh8idv\nMS12jmqdaOYTnJM1kFiddRvdkfChADq35x5bzV/oORYAmfurjuPN7ssHvrEEjmew\nbvi62MYtAkEAyDsAKrWsAfJQKbraTraJE7r7mTWxvAAYUONKKPZV2BXPzrTD/WMI\nn7z95pUu8x7anck9qqF6RYplo4fFLQKh/QJBANYwsszgGix33WUUbFwFAHFGN/40\n7CkwM/DhXW+mgS768jXNKSxDOS9MRSA1HbCMm5C2cw3Hcq9ULpUjyXeq7+kCQDx1\nvFYpJzgrP9Np7XNpILkJc+FOWk2nRbBfAUyfHUqzQ11qLef8GGWLfqs6jsOwpFiS\npIE6Yx5ObORVIc+2hM0CQE/pVhPEZ3boB8xoc9+3YL+++0yR2uMHoTY/q6r96kPC\n6C1oSRcDX/MUDOzC5HCUuwTYhNoN3FYkB5fov32BUbQ=\n-----END RSA PRIVATE KEY-----\n",
>   "client_email": "none",
>   "client_id": "114469363779822440226",
>   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
>   "token_uri": "https://oauth2.googleapis.com/token",
>   "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
>   "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/vertex%40tensorzero-public.iam.gserviceaccount.com",
>   "universe_domain": "googleapis.com"
> }
> ```
>
> This workflow is quite cumbersome, so we're planning to streamline it in the future (see [#575](https://github.com/tensorzero/tensorzero/issues/575)).

3. Launch the gateway in testing mode

   ```bash
   cargo run-e2e
   ```

> [!TIP]
>
> You can run a subset of tests with `cargo test-e2e xyz`, which will only run tests with `xyz` in their name.

4. Run the E2E tests
   ```bash
   cargo test-e2e
   ```

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
   uv pip install mypy
   uv run mypy . --strict
   ```

6. Run the formatter

   ```bash
   uv run ruff format --check .
   uv run ruff check --output-format=github --extend-select I .
   ```

---

Thanks again for your interest in contributing to TensorZero! We're excited to see what you build.
