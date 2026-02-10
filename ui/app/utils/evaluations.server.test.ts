import { describe, expect, test } from "vitest";
import {
  runEvaluation,
  killEvaluation,
  getRunningEvaluation,
} from "./evaluations.server";

describe("killEvaluation", () => {
  test("should return not-found for unknown evaluation run ID", () => {
    const result = killEvaluation("nonexistent-evaluation-id");
    expect(result).toEqual({
      killed: false,
      already_completed: false,
    });
  });

  test("should kill a running evaluation and mark it as completed", async () => {
    // Launch a real evaluation against the gateway (uses cached results by default)
    const startInfo = await runEvaluation(
      "entity_extraction",
      "foo",
      "gpt4o_mini_initial_prompt",
      /* concurrency */ 1,
      /* inferenceCache */ "on",
    );

    expect(startInfo.evaluation_run_id).toBeTruthy();
    expect(startInfo.num_datapoints).toBeGreaterThan(0);

    // The evaluation may or may not have completed already (cached results are fast).
    // Either way, killEvaluation should succeed.
    const killResult = killEvaluation(startInfo.evaluation_run_id);

    if (killResult.already_completed) {
      // Evaluation completed before we could kill it — that's fine with cached results
      expect(killResult).toEqual({ killed: false, already_completed: true });
    } else {
      // We successfully killed it
      expect(killResult).toEqual({ killed: true, already_completed: false });
    }

    // After killing, the evaluation should be marked as completed and killed
    const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
    expect(evaluation).toBeDefined();
    expect(evaluation?.completed).toBeInstanceOf(Date);
    if (killResult.killed) {
      expect(
        evaluation?.killed,
        "Killed evaluation should have killed flag set",
      ).toBe(true);
    }
  });

  test("should return already_completed when killing a completed evaluation", async () => {
    // Launch a small evaluation and wait for it to complete
    const startInfo = await runEvaluation(
      "entity_extraction",
      "foo",
      "gpt4o_mini_initial_prompt",
      /* concurrency */ 5,
      /* inferenceCache */ "on",
    );

    // Wait for completion (poll getRunningEvaluation)
    const maxWaitMs = 30_000;
    const pollIntervalMs = 100;
    let elapsed = 0;
    while (elapsed < maxWaitMs) {
      const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
      if (evaluation?.completed) break;
      await new Promise((r) => setTimeout(r, pollIntervalMs));
      elapsed += pollIntervalMs;
    }

    const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
    expect(
      evaluation?.completed,
      "Evaluation should have completed within timeout",
    ).toBeInstanceOf(Date);

    // Now try to kill the already-completed evaluation
    const killResult = killEvaluation(startInfo.evaluation_run_id);
    expect(killResult).toEqual({ killed: false, already_completed: true });
  });

  test("should abort the AbortController signal when killing", async () => {
    const startInfo = await runEvaluation(
      "entity_extraction",
      "foo",
      "gpt4o_mini_initial_prompt",
      /* concurrency */ 1,
      /* inferenceCache */ "on",
    );

    const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
    expect(evaluation).toBeDefined();

    // If the evaluation hasn't completed yet, kill it and check the signal
    if (!evaluation?.completed) {
      const abortSignal = evaluation!.abortController.signal;
      expect(abortSignal.aborted).toBe(false);

      killEvaluation(startInfo.evaluation_run_id);

      expect(abortSignal.aborted).toBe(true);
    }
    // If already completed (cached), the abort controller signal state doesn't matter
    // since the evaluation finished normally
  });
});
