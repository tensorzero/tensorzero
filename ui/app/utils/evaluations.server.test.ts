import { describe, expect, test } from "vitest";
import {
  cancelEvaluation,
  getRunningEvaluation,
  _test_registerRunningEvaluation,
} from "./evaluations.server";

describe("cancelEvaluation", () => {
  test("should return false for unknown evaluation run ID", () => {
    expect(cancelEvaluation("nonexistent-evaluation-id")).toBe(false);
  });

  test("should abort the controller and mark as cancelled", () => {
    const abortController = new AbortController();
    _test_registerRunningEvaluation("run-1", abortController);

    expect(
      abortController.signal.aborted,
      "Signal should not be aborted before cancel",
    ).toBe(false);

    expect(cancelEvaluation("run-1")).toBe(true);

    expect(
      abortController.signal.aborted,
      "Signal should be aborted after cancel",
    ).toBe(true);

    const evaluation = getRunningEvaluation("run-1");
    expect(evaluation?.completed).toBeInstanceOf(Date);
    expect(evaluation?.cancelled).toBe(true);
  });

  test("should not abort a naturally completed evaluation but should mark as cancelled", () => {
    const abortController = new AbortController();
    _test_registerRunningEvaluation("run-2", abortController, {
      completed: new Date(),
    });

    expect(cancelEvaluation("run-2")).toBe(true);

    expect(
      abortController.signal.aborted,
      "Should not abort a naturally completed evaluation",
    ).toBe(false);

    const evaluation = getRunningEvaluation("run-2");
    expect(
      evaluation?.cancelled,
      "Should still set cancelled flag to bypass grace period",
    ).toBe(true);
  });
});
