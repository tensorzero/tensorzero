import { describe, test, expect } from "vitest";
import {
  countTotalDynamicEvaluationRuns as countDynamicEvaluationRuns,
  getDynamicEvaluationRuns,
} from "./dynamic_evaluations.server";

describe("getDynamicEvaluationRuns", () => {
  test("should return correct run infos for", async () => {
    const runInfos = await getDynamicEvaluationRuns(10, 0);
    expect(runInfos).toMatchObject([
      {
        id: "0196880d-5d16-7b20-98a0-6570b7ca246d",
        name: "gpt-4.1-nano",
        project_name: "21_questions",
        tags: {},
        variant_pins: {
          ask_question: "gpt-4.1-nano",
        },
        timestamp: "2025-04-30T18:54:59Z",
      },
      {
        id: "0196880a-d42c-79e1-b2e1-f16e45477db5",
        name: "gpt-4.1-mini",
        project_name: "21_questions",
        tags: {},
        variant_pins: {
          ask_question: "gpt-4.1-mini",
        },
        timestamp: "2025-04-30T18:52:13Z",
      },
      {
        id: "01968806-6f22-77d1-bfd6-6f83df00b5ad",
        name: "baseline",
        project_name: "21_questions",
        tags: {},
        variant_pins: {
          ask_question: "baseline",
        },
        timestamp: "2025-04-30T18:47:25Z",
      },
    ]);
  });
});

describe("countDynamicEvaluationRuns", () => {
  test("should return correct total number of runs", async () => {
    const totalRuns = await countDynamicEvaluationRuns();
    expect(totalRuns).toBe(3);
  });
});
