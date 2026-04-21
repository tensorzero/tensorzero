// TypeScript judge evaluator that deliberately throws.
// Used in evaluations e2e tests to verify that a runtime exception in user
// code surfaces as an evaluator error (and feedback is NOT written).
function tensorzero_evaluator(
    _input: Input,
    _output: ContentBlockChatOutput[],
): boolean {
    throw new Error("tensorzero-evaluator-test: deliberate failure");
}
