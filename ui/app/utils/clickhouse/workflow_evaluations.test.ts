// NOTE: Test fixtures use the historical "tensorzero::dynamic_evaluation_run_id" tag name.
// The gateway now double-writes both the old tag name (for backward compatibility) and
// the new "tensorzero::workflow_evaluation_run_id" tag. All queries and UI components
// continue to use the old tag name. A future migration will update queries and backfill data.

import { describe, test, expect } from "vitest";
import {
  countWorkflowEvaluationRunEpisodes,
  getWorkflowEvaluationRunEpisodesByTaskName,
  getWorkflowEvaluationRunEpisodesByRunIdWithFeedback,
  getWorkflowEvaluationRunsByIds,
  getWorkflowEvaluationRunStatisticsByMetricName,
} from "./workflow_evaluations.server";

describe("getWorkflowEvaluationRunsByIds", () => {
  test("should return correct run infos for a given run id", async () => {
    const runInfos = await getWorkflowEvaluationRunsByIds([
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
      "01968d05-d734-7751-ab33-75dd8b3fb4a3",
    ]);
    expect(runInfos).toMatchObject([
      {
        id: "01968d05-d734-7751-ab33-75dd8b3fb4a3",
        name: "baseline",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:04:52Z",
        variant_pins: {
          answer_question: "baseline",
          ask_question: "baseline",
        },
      },
      {
        id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        name: "gpt-4.1-nano",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:02:56Z",
        variant_pins: {
          ask_question: "gpt-4.1-nano",
          answer_question: "baseline",
        },
      },
    ]);
  });
});

describe("getWorkflowEvaluationRunEpisodesByRunId", () => {
  test("should return correct episodes for a given run id", async () => {
    const episodes = await getWorkflowEvaluationRunEpisodesByRunIdWithFeedback(
      3,
      0,
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(episodes).toMatchObject([
      {
        episode_id: "0aaedb76-b457-7eae-aa62-145b73aa3e24",
        timestamp: "2025-05-01T18:02:56Z",
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        feedback_metric_names: ["elapsed_ms", "solved"],
        feedback_values: ["105946.19", "false"],
      },
      {
        episode_id: "0aaedb76-b457-7c86-8a61-2ffa01519447",
        timestamp: "2025-05-01T18:02:57Z",
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        feedback_metric_names: ["elapsed_ms", "goated", "solved"],
        feedback_values: ["106165", "true", "false"],
      },
      {
        episode_id: "0aaedb76-b457-7b77-ba8e-e395ed2f2218",
        timestamp: "2025-05-01T18:02:56Z",
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        feedback_metric_names: ["elapsed_ms", "solved"],
        feedback_values: ["46032.402", "true"],
      },
    ]);
    // TODO: add multiple (ragged) metrics, test that this is sorted by metric name
    // also test examples with no feedback and make sure the arrays are empty
  });
  test("should return correct episodes for a given run id", async () => {
    const episodes = await getWorkflowEvaluationRunEpisodesByRunIdWithFeedback(
      3,
      5,
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(episodes).toMatchObject([
      {
        episode_id: "0aaedb76-b456-70ef-b2ab-f844a165a25c",
        feedback_metric_names: ["elapsed_ms", "solved"],
        feedback_values: ["111887.65", "false"],
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        timestamp: "2025-05-01T18:02:57Z",
      },
      {
        episode_id: "0aaedb76-b452-7495-8b91-21df3115432d",
        feedback_metric_names: ["elapsed_ms", "solved"],
        feedback_values: ["86833.17", "true"],
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        timestamp: "2025-05-01T18:02:56Z",
      },
      {
        episode_id: "0aaedb76-b44e-7d60-a4f9-ad631caa4404",
        feedback_metric_names: ["elapsed_ms", "solved"],
        feedback_values: ["105674.7", "false"],
        run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        tags: {
          "tensorzero::dynamic_evaluation_run_id":
            "01968d04-142c-7e53-8ea7-3a3255b518dc",
        },
        task_name: null,
        timestamp: "2025-05-01T18:02:56Z",
      },
    ]);
  });
});

describe("getWorkflowEvaluationRunStatisticsByMetricName", () => {
  test("should return correct statistics for a given run id", async () => {
    const statistics = await getWorkflowEvaluationRunStatisticsByMetricName(
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    // The sort order of strings seems to be unstable on CI vs locally, so we made this test
    // order agnostic for now.
    const expected = [
      {
        metric_name: "elapsed_ms",
        count: 49,
        avg_metric: 91678.72114158163,
        stdev: 21054.80078125,
        ci_lower: 85783.37692283162,
        ci_upper: 97574.06536033163,
      },
      {
        metric_name: "goated",
        count: 1,
        avg_metric: 1,
        stdev: null,
        ci_lower: 0.20654329147389294,
        ci_upper: 1,
      },
      {
        metric_name: "solved",
        count: 49,
        avg_metric: 0.4489795918367347,
        stdev: 0.5025445456953674,
        ci_lower: 0.31852624929636336,
        ci_upper: 0.5868513320032188,
      },
    ];

    expect(statistics).toHaveLength(expected.length);

    expected.forEach((exp) => {
      expect(statistics).toEqual(
        expect.arrayContaining([expect.objectContaining(exp)]),
      );
    });
  });
});

describe("countWorkflowEvaluationRunEpisodes", () => {
  test("should return correct number of episodes for a given run id", async () => {
    const count = await countWorkflowEvaluationRunEpisodes(
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(count).toBe(50);
  });
});

describe("getWorkflowEvaluationRunByTaskName", () => {
  test("should return correct run by datapoint name", async () => {
    const runs = await getWorkflowEvaluationRunEpisodesByTaskName(
      ["01968d04-142c-7e53-8ea7-3a3255b518dc"],
      2,
      0,
    );
    expect(runs).toMatchObject([
      [
        {
          episode_id: "0aaedb76-b456-70ef-b2ab-f844a165a25c",
          feedback_metric_names: ["elapsed_ms", "solved"],
          feedback_values: ["111887.65", "false"],
          run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
          tags: {
            baz: "bat",
            foo: "bar",
            "tensorzero::dynamic_evaluation_run_id":
              "01968d04-142c-7e53-8ea7-3a3255b518dc",
          },
          task_name: null,
          timestamp: "2025-05-01T18:02:57Z",
        },
      ],
      [
        {
          episode_id: "0aaedb76-b457-700d-a59a-907787a96515",
          feedback_metric_names: ["elapsed_ms", "solved"],
          feedback_values: ["105675.805", "false"],
          run_id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
          tags: {
            baz: "bat",
            foo: "bar",
            "tensorzero::dynamic_evaluation_run_id":
              "01968d04-142c-7e53-8ea7-3a3255b518dc",
          },
          task_name: null,
          timestamp: "2025-05-01T18:02:57Z",
        },
      ],
    ]);
  });
});
