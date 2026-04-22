// TypeScript judge evaluator that deliberately exhausts the V8 heap.
//
// Used in evaluations e2e tests to verify that an OOM inside user code is
// caught by the runtime's near-heap-limit callback, terminates the
// isolate, and surfaces as an evaluator error (feedback must NOT be
// written). The `TypescriptJudgeExecutor` caps the isolate heap at 10 MiB,
// so we allocate a chain of arrays whose combined retained size easily
// exceeds that — a loop rather than one big allocation so the OOM trips
// before any single `new Array(...)` succeeds.
function tensorzero_evaluator(
    _input: Input,
    _output: ContentBlockChatOutput[],
): boolean {
    const retained: number[][] = [];
    // Each chunk is ~8 MB on 64-bit V8 (1,000,000 × 8-byte tagged pointers).
    // Five such chunks retained in a parent array = ~40 MB, well past the
    // 10 MiB heap cap applied in `TypescriptJudgeExecutor`.
    for (let i = 0; i < 5; i++) {
        const chunk: number[] = new Array(1_000_000);
        for (let j = 0; j < chunk.length; j++) {
            chunk[j] = j;
        }
        retained.push(chunk);
    }
    // Touch `retained` so the compiler can't eliminate the allocation.
    return retained.length > 0;
}
