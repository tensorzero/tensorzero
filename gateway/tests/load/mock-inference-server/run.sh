SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo 'POST http://localhost:3030/openai/chat/completions' \
| vegeta attack \
    -header="Content-Type: application/json" \
    -body=$SCRIPT_DIR/body.json \
    -duration=10s \
    -rate=1000 \
| vegeta report
