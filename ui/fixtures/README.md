# TensorZero fixtures

## Pulling fixtures from S3

Our small & large fixtures are stored in an S3-compatible object store (currently Cloudflare R2)
They can be manually downloaded with `uv run ./download-fixtures.py`

## Writing new fixtures

Fixtures should _not_ be committed to the repository. Instead:

1. Add the new fixtures to `./s3-fixtures`
2. Run `./upload-fixtures.sh`
3. List the new fixtures files in `uv run ./download-fixtures.py`
