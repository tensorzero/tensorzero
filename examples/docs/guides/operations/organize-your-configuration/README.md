# Code Example: Organize your configuration

This folder contains the code for the [Guides » Operations » Organize your configuration](https://www.tensorzero.com/docs/operations/organize-your-configuration) page in the documentation.

The example shows how to:

- Split your configuration into multiple files that reference each other.
- Enable file system access for MiniJinja templates to reuse shared snippets.

## Running the Example

1. Launch the TensorZero Gateway, TensorZero UI, and ClickHouse with `docker compose up`.
2. Run an inference:
   ```bash
   curl -X POST "http://localhost:3000/inference" \
   -H "Content-Type: application/json" \
   -d '{
       "function_name": "functionA",
       "input": {
       "messages": [
           {
           "role": "user",
           "content": "Share a fun fact about artificial intelligence."
           }
       ]
       }
   }'
   ```
