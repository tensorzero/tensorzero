# TensorZero fixtures

Most of our fixtures are stored in this directory, with the exception of some large files.

## Pulling fixtures from R2

Our fixtures are stored in Cloudflare R2. There are two ways to download them:

### Local development (no credentials needed)

Set `TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS=1` when running docker compose:

```bash
TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS=1 docker compose -f docker-compose.yml up
```

This downloads fixtures via public HTTP URLs. You can also run the HTTP scripts directly:

- **Native+LZ4 files** (large tables): `uv run ./download-large-fixtures-http.py`
- **JSONL files** (small tables): `uv run ./download-small-fixtures-http.py`

### CI / With R2 credentials (faster)

Set `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` environment variables. This uses `s5cmd` for faster, more reliable downloads:

- **Native+LZ4 files** (large tables): `uv run ./download-large-fixtures.py`
- **JSONL files** (small tables): `uv run ./download-small-fixtures.py`

## Writing new fixtures

Large fixtures should _not_ be committed to the repository. Instead:

**For large fixture files (Parquet -> Native+LZ4):**

Parquet files are the developer-facing source format. Native+LZ4 files are used by CI for faster ClickHouse ingestion.

1. Add or update the Parquet files in `./large-fixtures/`
2. Run `./convert-parquet-to-native.sh` to generate `.native.lz4` files (requires `clickhouse-local`)
3. Run `./upload-large-fixtures.sh` to upload both formats to R2
4. List the new fixture files in `download_fixtures_consts.py`
5. If row counts changed, update the hardcoded counts in `check-fixtures.sh`

To get row counts for `check-fixtures.sh`:

```bash
for f in large-fixtures/*.parquet; do
  echo "$f: $(clickhouse-local --query "SELECT count() FROM file('$f', 'Parquet')")"
done
```

**For JSONL files:**

1. Update the file locally
2. Rename with version suffix (e.g., `model_inference_examples_v2.jsonl`)
3. Run `./upload-small-fixtures.sh` to upload
4. Update the filename mapping in `download-small-fixtures.py`

**Upload credentials:** Uploads require R2 credentials. Use a subshell to set the right env vars:

```bash
(unset AWS_SESSION_TOKEN AWS_CREDENTIAL_EXPIRATION; AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" ./upload-small-fixtures.sh)
```
