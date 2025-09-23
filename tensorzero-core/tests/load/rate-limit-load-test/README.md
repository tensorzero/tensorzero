# Rate Limiting Load Tests

This directory contains a configurable load testing binary for TensorZero's rate limiting functionality using the `rlt` crate. The test focuses on database-level performance of the PostgreSQL-based rate limiting implementation.

## Prerequisites

1. **Start PostgreSQL**: The test uses the same PostgreSQL instance as the e2e tests:
   ```bash
   docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up postgres --wait
   ```

2. **Environment Setup**: Set the database URL (optional, defaults to localhost):
   ```bash
   export DATABASE_URL="postgres://postgres:postgres@localhost:5432/tensorzero_e2e_tests"
   ```

## Running the Load Test

### Using the cargo alias (recommended):
```bash
# Basic test with default settings
cargo test-rate-limit-load --duration 30s --rps 100

# High contention test (single key, many concurrent requests)
cargo test-rate-limit-load --capacity 10000 --contention-keys 0 --tokens-per-request 50 --duration 30s --rps 500

# Low contention test (many keys, minimal contention)
cargo test-rate-limit-load --contention-keys 1000 --duration 30s --rps 500

# Medium contention with custom pool
cargo test-rate-limit-load --contention-keys 50 --pool-size 100 --duration 60s --rps 300

# Burst test with TUI visualization
cargo test-rate-limit-load --capacity 5000 --refill-amount 2000 --tokens-per-request 100 --duration 15s --rps 1000 --collector tui

# Performance test with high throughput
cargo test-rate-limit-load --duration 120s --rps 1000 --max-concurrent 100
```

### Using cargo run directly:
```bash
cargo run --package rate-limit-load-test -- --duration 30s --rps 100
```

### Get help on available options:
```bash
cargo test-rate-limit-load --help
```

## Configuration Options

### Rate Limiting Parameters
- `--capacity <N>`: Token bucket capacity (default: 1,000,000)
- `--refill-amount <N>`: Tokens added per second (default: 1,000)
- `--tokens-per-request <N>`: Tokens consumed per request (default: 10)

### Load Testing Parameters
- `--duration <DURATION>`: Test duration (e.g., `30s`, `2m`, `1h`)
- `--rps <RATE>`: Target requests per second (e.g., `100`, `500.5`)
- `--max-concurrent <N>`: Maximum concurrent requests
- `--warmup <DURATION>`: Warmup period before collecting metrics

### Contention Control
- `--contention-keys <N>`: Number of different rate limit keys
  - `0`: Single key (maximum contention)
  - `1-50`: Medium contention
  - `100+`: Low contention

### Database Settings
- `--pool-size <N>`: PostgreSQL connection pool size (default: 50)

### Output Options
- `--collector <TYPE>`: Output format (`tui` for interactive, `silent` for minimal)

## Test Scenarios

### 1. Baseline Performance
```bash
cargo test-rate-limit-load --duration 30s --rps 100 --max-concurrent 10
```
- **Purpose**: Establish baseline performance metrics
- **Expected**: High success rate, low latency

### 2. High Contention
```bash
cargo test-rate-limit-load --capacity 10000 --contention-keys 0 --tokens-per-request 50 --duration 30s --rps 500 --max-concurrent 50
```
- **Purpose**: Test behavior under high contention on a single rate limit
- **Expected**: Some rate limiting, graceful degradation

### 3. Low Contention
```bash
cargo test-rate-limit-load --contention-keys 1000 --duration 30s --rps 500 --max-concurrent 50
```
- **Purpose**: Test performance with minimal contention across many rate limits
- **Expected**: Very high success rate, low latency

### 4. Burst Traffic
```bash
cargo test-rate-limit-load --capacity 5000 --refill-amount 2000 --tokens-per-request 100 --duration 15s --rps 1000 --max-concurrent 100
```
- **Purpose**: Test handling of sudden traffic spikes
- **Expected**: Graceful degradation under burst load

### 5. Endurance Test
```bash
cargo test-rate-limit-load --duration 300s --rps 200 --max-concurrent 20
```
- **Purpose**: Test stability over extended periods
- **Expected**: Consistent performance over time

## Understanding Results

The tool provides comprehensive metrics including:
- **Requests/sec**: Actual throughput achieved
- **Success rate**: Percentage of non-rate-limited requests
- **Latency percentiles**: Response time distribution (50th, 95th, 99th)
- **Error breakdown**: Types and counts of failures
- **Resource usage**: Connection pool utilization

### With TUI Visualization
Use `--collector tui` for real-time graphs and metrics:
- Live request rate and latency charts
- Success/error rate monitoring
- Resource utilization graphs
- Percentile latency distribution

## Architecture

- **`main.rs`**: CLI application with argument parsing and benchmark setup
- **`benchmark.rs`**: `rlt::BenchSuite` implementation and rate limiting logic
- **Database**: Tests run directly against PostgreSQL stored procedures used by TensorZero

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running: `docker ps | grep postgres`
- Check connection: `psql $DATABASE_URL -c "SELECT 1"`
- Verify migrations: Database migrations run automatically via docker-compose

### Performance Issues
- Monitor PostgreSQL logs for slow queries
- Check system resources (CPU, memory, disk I/O)
- Adjust `--pool-size` based on database capacity
- Use `--collector tui` to monitor real-time metrics

### Rate Limiting Behavior
- Success rates depend on bucket configuration and load
- Higher contention (fewer keys) leads to more serialization overhead
- Burst tests intentionally trigger rate limiting to test graceful degradation
- Rate limiting is expected behavior, not a failure

## Comparison with POC

This implementation:
- Uses actual TensorZero rate limiting infrastructure
- Tests real PostgreSQL stored procedures
- Provides configurable scenarios via CLI
- Integrates with TensorZero's development workflow
- Maintains compatibility with the original POC's `rlt` usage patterns
