#!/bin/bash
curl -X POST http://localhost:3000/v1/inference \
     -H "Content-Type: application/json" \
     -d '{"model": "openai::gpt-4o-mini", "messages": [{"role": "user", "content": "Analyze this PDF."}], "pdf": {"base64_content": "$(base64 tests/fixtures/sample.pdf)"}}'
