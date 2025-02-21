# Release Guide

This guide documents the steps to release a new version of TensorZero .

## Versioning

TensorZero follows [CalVer](https://calver.org/) for versioning.
The format is `YYYY.MM.PATCH` (e.g. `2025.01.0`, `2025.01.1`).

> [!IMPORTANT]
> Make sure to update every instance of the version in the codebase.
>
> The version is referenced in multiple places, including the client's `pyproject.toml` and the gateway's `status.rs`.

## Python Client

The Python client is published to PyPI via GitHub Actions automatically when a new release is made.

## Gateway Docker Container

Before building the Docker container for the first time, you need to set up your container builder:

```bash
docker buildx create \
  --name container-builder \
  --driver docker-container \
  --use \
  --bootstrap
```

Every time you want to build the Docker container, you need to run from the root of the repository:

```bash
DOCKER_BUILDKIT=1 docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t tensorzero/gateway:latest \
  -t tensorzero/gateway:XXXX.XX.X \
  -f gateway/Dockerfile \
  --attest type=provenance,mode=max \
  --attest type=sbom \
  --push \
  .
```

> [!IMPORTANT]
> Make sure to replace the `XXXX.XX.X` placeholder with the actual version of the Docker container you are building.

## UI Docker Container

Before building the Docker container for the first time, you need to set up your container builder:

```bash
docker buildx create \
  --name container-builder \
  --driver docker-container \
  --use \
  --bootstrap
```

Every time you want to build the Docker container, you need to run from `ui/`:

```bash
DOCKER_BUILDKIT=1 docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t tensorzero/ui:latest \
  -t tensorzero/ui:XXXX.XX.X \
  -f Dockerfile \
  --attest type=provenance,mode=max \
  --attest type=sbom \
  --push \
  .
```

> [!IMPORTANT]
> Make sure to replace the `XXXX.XX.X` placeholder with the actual version of the Docker container you are building.

## Documentation

Make sure to merge every PR with the `merge-on-release` label.

> [!NOTE]
> The documentation is currently stored in a separate repository.
> We're planning to make those files open source in the near future.

## Release Notes

Make sure to tag a release in the GitHub repository.
