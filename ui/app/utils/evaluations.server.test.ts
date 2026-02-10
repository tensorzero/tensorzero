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

  test("should return already_completed for a finished evaluation", async () => {
    const startInfo = await runEvaluation(
      "entity_extraction",
      "foo",
      "gpt4o_mini_initial_prompt",
      /* concurrency */ 5,
      /* inferenceCache */ "on",
    );

    // Wait for the evaluation to complete
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

    const cancelResult = cancelEvaluation(startInfo.evaluation_run_id);
    expect(cancelResult).toEqual({
      cancelled: false,
      already_completed: true,
    });
  });
});
