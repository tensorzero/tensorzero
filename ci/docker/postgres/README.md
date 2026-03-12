# tensorzero/postgres

Postgres images bundling [pgvector](https://github.com/pgvector/pgvector) and [pg_cron](https://github.com/citusdata/pg_cron) for use in TensorZero examples and development.

## Published tags

| Tag                                                   | Postgres version |
| ----------------------------------------------------- | ---------------- |
| `tensorzero/postgres:18.3`, `tensorzero/postgres:18`  | 18.3             |
| `tensorzero/postgres:17.9`, `tensorzero/postgres:17`  | 17.9             |
| `tensorzero/postgres:16.13`, `tensorzero/postgres:16` | 16.13            |
| `tensorzero/postgres:15.17`, `tensorzero/postgres:15` | 15.17            |
| `tensorzero/postgres:14.22`, `tensorzero/postgres:14` | 14.22            |

All images are multi-arch (`linux/amd64` and `linux/arm64`).

## Building and pushing

Images are built locally and pushed to Docker Hub (not via CI). This requires:

- **`docker buildx`** with a builder that supports `linux/amd64,linux/arm64`. Create one with:
  ```bash
  docker buildx create --name multiarch --use
  docker buildx inspect --bootstrap
  ```
- **Docker Hub login**: `docker login`

Then run:

```bash
# All versions
./build-and-push.sh

# Single version (by major)
./build-and-push.sh 17
```

Multi-arch builds with `--push` are not stored locally — the images go directly to the registry. There is no way to push after a local-only build, so `--push` is always included in the script.

## Local testing

To build for your local architecture and test without pushing:

```bash
docker build --build-arg PG_MAJOR=17 -t tensorzero/postgres:local ci/docker/postgres
docker run --rm tensorzero/postgres:local
```
