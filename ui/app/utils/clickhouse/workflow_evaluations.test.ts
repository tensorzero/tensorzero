// NOTE: Test fixtures use the historical "tensorzero::dynamic_evaluation_run_id" tag name.
// The gateway now double-writes both the old tag name (for backward compatibility) and
// the new "tensorzero::workflow_evaluation_run_id" tag. All queries and UI components
// continue to use the old tag name. A future migration will update queries and backfill data.

import { describe, test, expect } from "vitest";
import {
  countWorkflowEvaluationRunEpisodes,
  getWorkflowEvaluationRunEpisodesByRunIdWithFeedback,
} from "./workflow_evaluations.server";

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

describe("countWorkflowEvaluationRunEpisodes", () => {
  test("should return correct number of episodes for a given run id", async () => {
    const count = await countWorkflowEvaluationRunEpisodes(
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(count).toBe(50);
  });
});
