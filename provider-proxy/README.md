# `provider-proxy`

> [!NOTE]
>
> **_This package is for people building TensorZero itself, not for people using TensorZero for their applications._**

This is a caching MITM proxy, used to cache (notoriously flaky) model provider requests for our E2E tests.
This is a caching MITM proxy, used to cache model provider requests during tensorzero e2e test

## Usage

The proxy can be started with `cargo run`. By default, it runs on port `3003` and writes cache entries to `./request_cache`.
Use `cargo run -- --help` for more information.

To use this proxy with the e2e tests, set `TENSORZERO_E2E_PROXY="http://localhost:3003"` when running e2e or batch tests
(e.g. `TENSORZERO_E2E_PROXY="http://localhost:3003" cargo run-e2e`)
