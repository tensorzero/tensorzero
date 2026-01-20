# TensorZero UI

The TensorZero UI provides a web interface to help manage your TensorZero deployments.
The UI provides functionality for observability, optimization, and more.

## Running the UI

The easiest way to run the UI is to use the `tensorzero/ui` Docker image.
See the [Quick Start](https://www.tensorzero.com/docs/quickstart/) and the [TensorZero UI Deployment Guide](https://www.tensorzero.com/docs/ui/deployment/) for more information.

## Development Setup

Follow the instructions from [`CONTRIBUTING.md`](../CONTRIBUTING.md#tensorzero-ui) to set up your development environment.

## Things to note

1. To test optimization workflows without real provider APIs, spin up the `mock-provider-api` and set `TENSORZERO_INTERNAL_MOCK_PROVIDER_API=http://localhost:3030` when running the gateway.
2. For any new code, prefer `undefined` over `null`. The only place to use `null` is for `napi-rs` compatibility, because it uses `null` to represent an `Option<T>`. Never write a type as `T | undefined | null`.
