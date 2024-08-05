# Load Testing

Before running a load test, launch the mock inference server and the gateway (on separate terminals):

```
cargo run --release --bin mock-inference-provider
```

```
cargo run --release --bin gateway
```

Then, you can run a load test with `sh path/to/test/run.sh`.
