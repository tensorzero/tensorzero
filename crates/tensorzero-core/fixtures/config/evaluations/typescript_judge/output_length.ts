// TypeScript judge evaluator that returns the character length of the
// model's first text content block (or 0 if there isn't one).
//
// Used in evaluations e2e tests to verify that the typescript judge
// receives the real inference output and can return a non-trivial float.
function tensorzero_evaluator({ output }: EvaluatorParams): number {
    for (const block of output) {
        if (block.type === "text") {
            return block.text.length;
        }
    }
    return 0;
}
