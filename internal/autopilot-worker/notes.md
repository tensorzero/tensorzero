I need to build a `durable` worker that is able to call tensorzero-core endpoints in `internal/autopilot-worker`. This should run as a long-lived task in the main body of the gateway alongside the axum server. It should have access to whatever state is needed to call all endpoints.

The worker will run tools from `autopilot-tools` that either directly call endpoints or that wrap functions that call endpoints using the state.
We should pass the state through durable's application state generic.

We will need to also extend the `ClientTool` traits to extend either `SimpleTool` or `TaskTool`.
We should then add a wrapper type that adds logic that wraps `impl ClientTool` and itself `impl`s `TaskTool` and uses some side info containing the tool call event id, tool call id, and `session_id` to send an event back to the autopilot API with the Tool result after the tool completes. This should itself use a ctx.step.
The tools crate should export a registry of tools that implement `ClientTool` that are wrapped by our wrapper somewhere.
The worker should then be a binary that executes the registry with `durable-tools`.
Please make a plan first, write it in a markdown file, and wait for review.
