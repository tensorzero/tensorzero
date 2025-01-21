# Mock Inference Provider

This is a mock inference provider that can be used to test the gateway.

## Usage

To run the mock inference provider, you can use the following command:

```bash
cargo run --profile performance --bin mock-inference-provider
```

By default, the mock inference provider will bind to `0.0.0.0:3030`.
You can optionally specify the address to bind to using the first CLI argument.
For example, to bind to `0.0.0.0:1234`, you'd run:

```bash
cargo run --profile performance --bin mock-inference-provider -- 0.0.0.0:1234
```
