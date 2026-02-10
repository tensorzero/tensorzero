import { describe, expect, test } from "vitest";
import {
  runEvaluation,
  cancelEvaluation,
  getRunningEvaluation,
} from "./evaluations.server";

describe("cancelEvaluation", () => {
  test("should return not-found for unknown evaluation run ID", () => {
    const result = cancelEvaluation("nonexistent-evaluation-id");
    expect(result).toEqual({
      cancelled: false,
      already_completed: false,
    });
  });

  test("should cancel a running evaluation and mark it as completed", async () => {
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
    // Either way, cancelEvaluation should succeed.
    const cancelResult = cancelEvaluation(startInfo.evaluation_run_id);

    if (cancelResult.already_completed) {
      // Evaluation completed before we could cancel it — that's fine with cached results
      expect(cancelResult).toEqual({
        cancelled: false,
        already_completed: true,
      });
    } else {
      // We successfully cancelled it
      expect(cancelResult).toEqual({
        cancelled: true,
        already_completed: false,
      });
    }

    // After cancelling, the evaluation should be marked as completed
    const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
    expect(evaluation).toBeDefined();
    expect(evaluation?.completed).toBeInstanceOf(Date);
    if (cancelResult.cancelled) {
      expect(
        evaluation?.cancelled,
        "Cancelled evaluation should have cancelled flag set",
      ).toBe(true);
    }
  });

  test("should return already_completed when cancelling a completed evaluation", async () => {
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

    // Now try to cancel the already-completed evaluation
    const cancelResult = cancelEvaluation(startInfo.evaluation_run_id);
    expect(cancelResult).toEqual({
      cancelled: false,
      already_completed: true,
    });
  });

  test("should not expose abortController through getRunningEvaluation", async () => {
    const startInfo = await runEvaluation(
      "entity_extraction",
      "foo",
      "gpt4o_mini_initial_prompt",
      /* concurrency */ 1,
      /* inferenceCache */ "on",
    );

    const evaluation = getRunningEvaluation(startInfo.evaluation_run_id);
    expect(evaluation).toBeDefined();
    expect(
      "abortController" in evaluation!,
      "abortController should not be exposed",
    ).toBe(false);
  });
});
