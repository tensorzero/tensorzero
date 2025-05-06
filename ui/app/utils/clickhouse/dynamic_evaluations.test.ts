import { describe, test, expect } from "vitest";
import {
  countDynamicEvaluationProjects,
  countDynamicEvaluationRunEpisodes,
  countDynamicEvaluationRuns as countDynamicEvaluationRuns,
  getDynamicEvaluationProjects,
  getDynamicEvaluationRunEpisodesByTaskName,
  getDynamicEvaluationRunEpisodesByRunIdWithFeedback,
  getDynamicEvaluationRuns,
  getDynamicEvaluationRunsByIds,
  getDynamicEvaluationRunStatisticsByMetricName,
  searchDynamicEvaluationRuns,
} from "./dynamic_evaluations.server";

describe("getDynamicEvaluationRuns", () => {
  test("should return correct run infos for", async () => {
    const runInfos = await getDynamicEvaluationRuns(10, 0);
    expect(runInfos).toMatchObject([
      {
        id: "0196a0e5-9600-7c83-ab3b-dac7598fddbd",
        name: "agent-gemini-2.5-flash-compact_context-baseline",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "baseline",
          multi_hop_rag_agent: "gemini-2.5-flash",
        },
      },
      {
        id: "0196a0e5-9600-7c83-ab3b-dabb145a9dbe",
        name: "agent-baseline-compact_context-baseline",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "baseline",
          multi_hop_rag_agent: "baseline",
        },
      },
      {
        id: "0196a0e5-9600-7c83-ab3b-daa3fd908279",
        name: "agent-baseline-compact_context-gemini-2.5-flash",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "gemini-2.5-flash",
          multi_hop_rag_agent: "baseline",
        },
      },
      {
        id: "0196a0e5-9600-7c83-ab3b-da910bfdd03e",
        name: "agent-gpt-4.1-mini-compact_context-baseline",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "baseline",
          multi_hop_rag_agent: "gpt-4.1-mini",
        },
      },
      {
        id: "0196a0e5-9600-7c83-ab3b-da81097b66cd",
        name: "agent-gemini-2.5-flash-compact_context-gemini-2.5-flash",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "gemini-2.5-flash",
          multi_hop_rag_agent: "gemini-2.5-flash",
        },
      },
      {
        id: "0196a0e5-9600-7c83-ab3b-da7bc6aac7e7",
        name: "agent-gpt-4.1-mini-compact_context-gemini-2.5-flash",
        project_name: "beerqa-agentic-rag",
        tags: {},
        timestamp: "2025-05-05T14:42:02Z",
        variant_pins: {
          compact_context: "gemini-2.5-flash",
          multi_hop_rag_agent: "gpt-4.1-mini",
        },
      },
      {
        id: "01968d08-3198-7e82-b3c8-e228f8ddc779",
        name: "gpt-4.1-mini",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:07:26Z",
        variant_pins: {
          ask_question: "gpt-4.1-mini",
          answer_question: "baseline",
        },
      },
      {
        id: "01968d05-d734-7751-ab33-75dd8b3fb4a3",
        name: "baseline",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:04:52Z",
        variant_pins: {
          ask_question: "baseline",
          answer_question: "baseline",
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

  test("should return correct run infos for a given run id", async () => {
    const runInfos = await getDynamicEvaluationRuns(
      10,
      0,
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(runInfos).toMatchObject([
      {
        id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        name: "gpt-4.1-nano",
        project_name: "21_questions",
        tags: {},
        timestamp: "2025-05-01T18:02:56Z",
        variant_pins: {
          ask_question: "gpt-4.1-nano",
        },
      },
    ]);
  });

  test("should return correct run infos for a given project name", async () => {
    const runInfos = await getDynamicEvaluationRuns(
      10,
      0,
      undefined,
      "21_questions",
    );
    expect(runInfos).toMatchObject([
      {
        id: "01968d08-3198-7e82-b3c8-e228f8ddc779",
        name: "gpt-4.1-mini",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:07:26Z",
        variant_pins: {
          answer_question: "baseline",
          ask_question: "gpt-4.1-mini",
        },
      },
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
          answer_question: "baseline",
          ask_question: "gpt-4.1-nano",
        },
      },
    ]);
  });
});

describe("getDynamicEvaluationRunsByIds", () => {
  test("should return correct run infos for a given run id", async () => {
    const runInfos = await getDynamicEvaluationRunsByIds([
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

describe("countDynamicEvaluationRuns", () => {
  test("should return correct total number of runs", async () => {
    const totalRuns = await countDynamicEvaluationRuns();
    expect(totalRuns).toBe(9);
  });
});

describe("getDynamicEvaluationRunEpisodesByRunId", () => {
  test("should return correct episodes for a given run id", async () => {
    const episodes = await getDynamicEvaluationRunEpisodesByRunIdWithFeedback(
      3,
      0,
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(episodes).toMatchObject([
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
    ]);
    // TODO: add multiple (ragged) metrics, test that this is sorted by metric name
    // also test examples with no feedback and make sure the arrays are empty
  });
  test("should return correct episodes for a given run id", async () => {
    const episodes = await getDynamicEvaluationRunEpisodesByRunIdWithFeedback(
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

describe("getDynamicEvaluationRunStatisticsByMetricName", () => {
  test("should return correct statistics for a given run id", async () => {
    const statistics = await getDynamicEvaluationRunStatisticsByMetricName(
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
        ci_error: 5895.3442187499995,
      },
      {
        metric_name: "goated",
        count: 1,
        avg_metric: 1,
        stdev: null,
        ci_error: null,
      },
      {
        metric_name: "solved",
        count: 49,
        avg_metric: 0.4489795918367347,
        stdev: 0.5025445456953674,
        ci_error: 0.14071247279470286,
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

describe("countDynamicEvaluationRunEpisodes", () => {
  test("should return correct number of episodes for a given run id", async () => {
    const count = await countDynamicEvaluationRunEpisodes(
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(count).toBe(50);
  });
});

describe("getDynamicEvaluationProjects", () => {
  test("should return correct projects", async () => {
    const projects = await getDynamicEvaluationProjects(10, 0);
    expect(projects).toMatchObject([
      {
        count: 6,
        last_updated: "2025-05-05T14:42:02Z",
        name: "beerqa-agentic-rag",
      },
      {
        count: 3,
        last_updated: "2025-05-01T18:07:26Z",
        name: "21_questions",
      },
    ]);
  });
});

describe("countDynamicEvaluationProjects", () => {
  test("should return correct number of projects", async () => {
    const count = await countDynamicEvaluationProjects();
    expect(count).toBe(2);
  });
});

describe("searchDynamicEvaluationRuns", () => {
  test("should return correct runs by display name with project name set", async () => {
    const runs = await searchDynamicEvaluationRuns(
      10,
      0,
      "21_questions",
      "gpt",
    );
    expect(runs).toMatchObject([
      {
        id: "01968d08-3198-7e82-b3c8-e228f8ddc779",
        name: "gpt-4.1-mini",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:07:26Z",
        variant_pins: {
          ask_question: "gpt-4.1-mini",
          answer_question: "baseline",
        },
      },
      {
        id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        name: "gpt-4.1-nano",
        project_name: "21_questions",
        tags: { foo: "bar" },
        timestamp: "2025-05-01T18:02:56Z",
        variant_pins: {
          answer_question: "baseline",
          ask_question: "gpt-4.1-nano",
        },
      },
    ]);
  });

  test("should return correct runs by run id without project name set", async () => {
    const runs = await searchDynamicEvaluationRuns(
      10,
      0,
      undefined,
      "01968d04-142c-7e53-8ea7-3a3255b518dc",
    );
    expect(runs).toMatchObject([
      {
        id: "01968d04-142c-7e53-8ea7-3a3255b518dc",
        name: "gpt-4.1-nano",
      },
    ]);
  });
});

describe("getDynamicEvaluationRunByTaskName", () => {
  test("should return correct run by datapoint name", async () => {
    const runs = await getDynamicEvaluationRunEpisodesByTaskName(
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
