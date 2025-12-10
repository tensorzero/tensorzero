import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TensorZeroClient } from "~/utils/tensorzero/tensorzero";

let tensorZeroClient: TensorZeroClient;

describe("update datapoints", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should preserve source_inference_id, generate a new ID, and set is_custom to true when updating a datapoint", async () => {
    // Create datapoint from inference
    const inferenceId = "0196368e-5505-7721-88d2-644a2da892a7";
    const createResult =
      await tensorZeroClient.createDatapointFromInferenceLegacy(
        "test",
        inferenceId,
        "inherit",
        "extract_entities",
        "gpt4o_initial_prompt",
        "0196368e-5505-7721-88d2-645619b42142",
      );

    // Verify initial state: is_custom should be false, source_inference_id should match
    const initialDatapoint = await tensorZeroClient.getDatapoint(
      createResult.id,
      /*datasetName=*/ "test",
    );
    expect(initialDatapoint).toBeDefined();
    expect(initialDatapoint?.is_custom).toBe(false);
    expect(initialDatapoint?.source_inference_id).toBe(inferenceId);

    // TypeScript refinement: we've verified initialDatapoint is defined
    if (!initialDatapoint || initialDatapoint.type !== "json") {
      throw new Error("Expected JSON datapoint");
    }

    // Update the datapoint (e.g., modify the output)
    const updatedOutput = {
      person: ["Updated Person"],
      organization: ["Updated Org"],
      location: ["Updated Location"],
      miscellaneous: ["Updated Misc"],
    };

    const updateResult = await tensorZeroClient.updateDatapoint("test", {
      type: "json",
      id: initialDatapoint.id,
      input: initialDatapoint.input,
      output: {
        raw: JSON.stringify(updatedOutput),
      },
      output_schema: initialDatapoint.output_schema,
    });

    // Verify updated state
    const updatedDatapoint = await tensorZeroClient.getDatapoint(
      updateResult.id,
      /*datasetName=*/ "test",
    );

    // New ID should be created
    expect(updateResult.id).not.toBe(createResult.id);

    // source_inference_id should be preserved
    expect(updatedDatapoint?.source_inference_id).toBe(inferenceId);

    // is_custom should now be true (custom modification)
    expect(updatedDatapoint?.is_custom).toBe(true);
  });

  test("should list datapoints", async () => {
    const datapoints = await tensorZeroClient.listDatapoints("foo", {
      function_name: "extract_entities",
      limit: 10,
      offset: 0,
    });
    expect(datapoints.datapoints.length).toBe(10);
    for (const datapoint of datapoints.datapoints) {
      expect(datapoint.function_name).toBe("extract_entities");
    }
  });
});

describe("getInferenceStats", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should return inference count for a function", async () => {
    const stats = await tensorZeroClient.getInferenceStats("extract_entities");
    expect(stats.inference_count).toBeGreaterThanOrEqual(604);
  });

  test("should return inference count for a function and variant", async () => {
    const stats = await tensorZeroClient.getInferenceStats(
      "extract_entities",
      "gpt4o_initial_prompt",
    );
    expect(stats.inference_count).toBeGreaterThanOrEqual(132);
  });

  test("should throw error for unknown function", async () => {
    await expect(
      tensorZeroClient.getInferenceStats("nonexistent_function"),
    ).rejects.toThrow();
  });

  test("should throw error for unknown variant", async () => {
    await expect(
      tensorZeroClient.getInferenceStats(
        "extract_entities",
        "nonexistent_variant",
      ),
    ).rejects.toThrow();
  });
});

describe("getFeedbackStats", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should return feedback stats for boolean metric", async () => {
    const stats = await tensorZeroClient.getFeedbackStats(
      "extract_entities",
      "exact_match",
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(99);
    expect(stats.inference_count).toBeGreaterThanOrEqual(41);
  });

  test("should return feedback stats for float metric with threshold", async () => {
    const stats = await tensorZeroClient.getFeedbackStats(
      "extract_entities",
      "jaccard_similarity",
      0.8,
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(99);
    expect(stats.inference_count).toBeGreaterThanOrEqual(54);
  });

  test("should return feedback stats for demonstration metric", async () => {
    const stats = await tensorZeroClient.getFeedbackStats(
      "extract_entities",
      "demonstration",
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(100);
    // For demonstrations, feedback_count equals inference_count
    expect(stats.inference_count).toBe(stats.feedback_count);
  });

  test("should throw error for unknown function", async () => {
    await expect(
      tensorZeroClient.getFeedbackStats("nonexistent_function", "exact_match"),
    ).rejects.toThrow();
  });

  test("should throw error for unknown metric", async () => {
    await expect(
      tensorZeroClient.getFeedbackStats(
        "extract_entities",
        "nonexistent_metric",
      ),
    ).rejects.toThrow();
  });
});
