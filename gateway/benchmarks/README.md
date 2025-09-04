# Benchmarks

## TensorZero Gateway vs. LiteLLM Proxy (LiteLLM Gateway)

### Environment Setup

- Launch an AWS EC2 Instance: `c7i.xlarge` (4 vCPUs, 8 GB RAM)
- Increase the limits for open file descriptors:
  - Run `sudo vim /etc/security/limits.conf` and add the following lines:
    ```
    *               soft    nofile          65536
    *               hard    nofile          65536
    ```
  - Run `sudo vim /etc/pam.d/common-session` and add the following line:
    ```
    session required pam_limits.so
    ```
  - Reboot the instance with `sudo reboot`
  - Run `ulimit -Hn` and `ulimit -Sn` to check that the limits are now `65536`

- Install Python 3.10.14.
- Install LiteLLM: `pip install 'litellm[proxy]'==1.74.9`
- Install Rust 1.80.1.
- Install `vegeta` [→](https://github.com/tsenart/vegeta).
- Set the `OPENAI_API_KEY` environment variable to anything (e.g. `OPENAI_API_KEY=test`).

### Test Setup

- Launch the mock inference provider in performance mode:

  ```bash
  cargo run --profile performance --bin mock-inference-provider
  ```

#### TensorZero Gateway

- Launch the TensorZero Gateway in performance mode (without observability):

  ```bash
  cargo run --profile performance --bin gateway tensorzero-core/tests/load/tensorzero-without-observability.toml
  ```

- Run the benchmark:
  ```bash
  sh tensorzero-core/tests/load/simple/run.sh
  ```

#### LiteLLM Gateway (LiteLLM Proxy)

- Launch the LiteLLM Gateway:

  ```
  litellm --config tensorzero-core/tests/load/simple-litellm/config.yaml --num_workers=4
  ```

- Run the benchmark:

  ```bash
  sh tensorzero-core/tests/load/simple-litellm/run.sh
  ```
