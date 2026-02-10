import { describe, expect, test } from "vitest";
import {
  cancelEvaluation,
  getRunningEvaluation,
  _test_registerRunningEvaluation,
} from "./evaluations.server";

describe("cancelEvaluation", () => {
  test("should return not-found for unknown evaluation run ID", () => {
    const result = cancelEvaluation("nonexistent-evaluation-id");
    expect(result).toEqual({
      cancelled: false,
      already_completed: false,
    });
  });

  test("should abort the controller, mark completed, and set cancelled flag", () => {
    const abortController = new AbortController();
    _test_registerRunningEvaluation("run-1", abortController);

    expect(
      abortController.signal.aborted,
      "Signal should not be aborted before cancel",
    ).toBe(false);

    const result = cancelEvaluation("run-1");
    expect(result).toEqual({ cancelled: true, already_completed: false });

    expect(
      abortController.signal.aborted,
      "Signal should be aborted after cancel",
    ).toBe(true);

    const evaluation = getRunningEvaluation("run-1");
    expect(evaluation).toBeDefined();
    expect(evaluation?.completed).toBeInstanceOf(Date);
    expect(evaluation?.cancelled).toBe(true);
  });

  test("should return already_completed and not abort if evaluation finished naturally", () => {
    const abortController = new AbortController();
    _test_registerRunningEvaluation("run-2", abortController, {
      completed: new Date(),
    });

    const result = cancelEvaluation("run-2");
    expect(result).toEqual({ cancelled: false, already_completed: true });

    expect(
      abortController.signal.aborted,
      "Should not abort a naturally completed evaluation",
    ).toBe(false);
  });
});
