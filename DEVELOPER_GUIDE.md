# Developer Guide [WIP]

> [!IMPORTANT]
>
> **This guide is for developers planning to contribute to TensorZero, not for developers planning to use TensorZero.**

## Tests

### Rust

#### Unit Tests

```bash
cargo test-unit
```

#### E2E Tests

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
> For GCP Vertex AI, you'll need a credentials file. You can use the following fake file:
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

3. Launch the gateway in testing mode

   ```bash
   cargo run-e2e
   ```

> [!TIP]
>
> You can run a subset of tests with `cargo test-e2 xyz`, which will only run tests with `xyz` in their name.

4. Run the E2E tests
   ```bash
   cargo test-e2e
   ```
