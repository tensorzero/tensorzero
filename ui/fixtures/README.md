# TensorZero fixtures

Most of our fixtures are stored in this directory, with the exception of some large files.

## Pulling fixtures from R2

Our large fixtures are stored in Cloudflare R2:

- **Parquet files** (large tables): `uv run ./download-large-fixtures.py`
- **JSONL files** (small tables): `uv run ./download-small-fixtures.py`

Downloads use public URLs by default. In CI, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` env vars enable authenticated access.

## Writing new fixtures

Large fixtures should _not_ be committed to the repository. Instead:

**For parquet files:**

1. Add the new fixtures to `./s3-fixtures`
2. Run `./upload-large-fixtures.sh`
3. List the new fixture files in `download-large-fixtures.py`

**For JSONL files:**

1. Update the file locally
2. Rename with version suffix (e.g., `model_inference_examples_v2.jsonl`)
3. Run `./upload-small-fixtures.sh` to upload
4. Update the filename mapping in `download-small-fixtures.py`

**Upload credentials:** Uploads require R2 credentials. Use a subshell to set the right env vars:

```bash
(unset AWS_SESSION_TOKEN AWS_CREDENTIAL_EXPIRATION; AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" ./upload-small-fixtures.sh)
```
