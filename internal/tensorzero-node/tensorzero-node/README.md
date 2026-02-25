# TensorZero Internal Node Client

This directory contains initial efforts at Node bindings for TensorZero via [NAPI-RS](https://napi.rs/).
For now, it contains methods for starting and polling optimization jobs and for getting config once parsed.
The functionality is exposed by a class TensorZeroClient that provides these methods.
All Rust bindings to this point take strings on the way in and return strings as well but we build bindings that show what types these strings should be so the calling code can interface with them in a type-safe way.

## Building

You can build the Rust code for this library with `pnpm build` from this directory or `pnpm -r build` from anywhere in the repository.
You **must** build the bindings if they have changed using `pnpm build-bindings` from this directory or CI will not accept your PR.

## Testing

We have unit tests in this directory for some of the client's functionality and that test that the bindings work.
These can be run via `pnpm test`.
We also have e2e tests that test this client in a broader context in the `ui/` directory.
