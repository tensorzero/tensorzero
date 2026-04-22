// TypeScript judge evaluator that unconditionally returns true.
// Used in evaluations e2e tests to verify the happy path: the typescript
// code compiles, runs, calls FINAL(true), and feedback is recorded.
function tensorzero_evaluator(_params: EvaluatorParams): boolean {
    return true;
}
