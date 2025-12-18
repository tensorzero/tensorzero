# Buildkite CI scripts

As we scale TensorZero, we'll want CI processes that are portable, reliable, and scalable.
This directory contains scripts for running CI jobs on Buildkite.

The high-level architecture of CI here is that there are 2 workflows, `pr-tests.yml` and `merge-queue-tests.yml`.
Each builds a set of Docker images for testing for stuff like the gateway, ui, test runners, and even the gateway with e2e fixtures.
These are pushed to Docker Hub and pulled by the CI jobs that actually run the tests.
The tests then export their results to Buildkite's managed test platform.

Check out the `build-*-container.sh` scripts to see how we build the images, and the `*-tests.sh` scripts to see the pattern for running tests.
In the normal course of development, tests can be run locally using the typical `cargo test-unit`, `cargo test-e2e`, `pnpm test`, and `pnpm test-e2e` commands but in CI we build images containing all the dependencies and enforce that all communication between services is done by Docker compose networking.
This sets us up nicely for eventually testing with K8s and for future tests of portable deployments of TensorZero.
We use retries on image building to help account for issues with dependencies.

## Roadmap

The goal in the near term is to replace our current GitHub actions workflows with Buildkite workflows for all nontrivial tests we run.
The remaining tests to be ported over are:

- Tests of our python client
- Tests of various OpenAI clients
- Replicated and cloud clickhouse testing (the latter is already on buildkite but doesn't fit the current pattern)
- Batch tests
- E2E tests that rebuild the model inference cache and make sure they can re-run without credentials.
- Publish as public images on `tensorzero-internal/` on Docker Hub and make the images public so we don't need creds to pul.
- Deduplicate the `build-*.sh` scripts.

We'll also want to swap out the current GitHub actions workflows with BuildKite webhooks since these seem better integrated.
Once we reach this point we could potentially rip the existing CI workflow almost entirely (probably can and should leave lints and stuff on GHA).

## Running locally

To run these tests locally, you can do something like `docker compose -f tensorzero-core/fixtures/docker-compose.live.yml build` and then `docker compose -f tensorzero-core/fixtures/docker-compose.live.yml run --rm live-tests`. This will be a bit slower than the native workflow (and takes ~15 minutes to build on a new MacBook Pro) but it is extremely useful to replicate CI test conditions on a local machine.

## BuildKite MCP

One super neat feature of buildkite is the open-souce [MCP server](https://github.com/buildkite/buildkite-mcp-server) they offer. Once you install this you can run a command like `claude mcp add buildkite --env BUILDKITE_API_TOKEN=bkua_xxxxxxxx -- buildkite-mcp-server stdio` to give Claude access to the MCP server.
You can use it to ask the status of jobs and also why they're failing etc.

## XUnit viewer

[Xunit viewer](https://www.npmjs.com/package/xunit-viewer) is a neat utility for generating HTML reports from JUnit XML files. It can be used to visualize the results of your tests and identify areas for improvement.

`xunit-viewer -r target/nextest/e2e/junit.xml` to run on an example JUnit.

## Areas for incremental improvement

- improve build caching locally
- use remote Docker builder for better caching and faster container builds
- work on limiting the size of test docker images, they are way too big
- make sure the e2e tests also map the JUnit files to the local filesystem so they can be exported.
