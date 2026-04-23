// TypeScript judge evaluator that returns the character length of the first
// text content block in `reference_output`, or -1 if the datapoint has no
// reference output.
//
// Used in evaluations e2e tests to verify that the datapoint's expected
// output is plumbed through to the typescript judge as `reference_output`.
function tensorzero_evaluator({ reference_output }: EvaluatorParams): number {
    if (!reference_output) {
        return -1;
    }
    for (const block of reference_output) {
        if (block.type === "text") {
            return block.text.length;
        }
    }
    return 0;
}
