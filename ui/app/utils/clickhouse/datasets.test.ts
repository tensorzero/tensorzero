import { DatasetQueryParamsSchema } from "./datasets";
import { countRowsForDataset } from "./datasets.server";
import { expect, test, describe } from "vitest";

describe("countRowsForDataset", () => {
  test("returns the correct number of rows for a specific function", async () => {
    const dataset_params = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      output_source: "none",
    });
    const rows = await countRowsForDataset(dataset_params);
    expect(rows).toBe(494);
  });

  test("returns the correct number of rows for a specific variant", async () => {
    const dataset_params = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      variant_name: "llama_8b_initial_prompt",
      output_source: "none",
    });
    const rows = await countRowsForDataset(dataset_params);
    expect(rows).toBe(110);
  });

  test("throws an error if function_name is not provided but variant_name is", async () => {
    await expect(async () => {
      const dataset_params = DatasetQueryParamsSchema.parse({
        inferenceType: "chat",
        variant_name: "test",
        output_source: "none",
      });
      await countRowsForDataset(dataset_params);
    }).rejects.toThrow(
      "If variant_name is provided, function_name must also be provided.",
    );
  });

  test("returns the correct number of rows when filtering by a specific metric", async () => {
    const dataset_params = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_rating",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "id",
      },
      output_source: "none",
    });
    const rows = await countRowsForDataset(dataset_params);
    expect(rows).toBe(67);
  });

  test("returns correct count for boolean metrics at inference level", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "exact_match",
        metric_type: "boolean",
        operator: ">",
        threshold: 0,
        join_on: "id",
      },
      output_source: "inference",
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(41);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_score",
        metric_type: "boolean",
        operator: ">",
        threshold: 0,
        join_on: "id",
      },
      output_source: "none",
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(80);
  });

  test("returns correct count for boolean metrics at episode level", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "exact_match_episode",
        metric_type: "boolean",
        operator: ">",
        threshold: 0,
        join_on: "episode_id",
      },
      output_source: "inference",
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(29);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_score_episode",
        metric_type: "boolean",
        operator: ">",
        threshold: 0,
        join_on: "episode_id",
      },
      output_source: "none",
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(9);
  });

  test("returns correct count for float metrics at inference level", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "jaccard_similarity",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "id",
      },
      output_source: "none",
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(54);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_rating",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "id",
      },
      output_source: "none",
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(67);
  });

  test("returns correct count for float metrics at episode level", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "jaccard_similarity_episode",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "episode_id",
      },
      output_source: "none",
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(35);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_rating_episode",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "episode_id",
      },
      output_source: "none",
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(11);
  });

  test("returns correct count for demonstration metrics", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      output_source: "demonstration",
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(100);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      output_source: "demonstration",
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(493);
  });

  test("returns correct count for rows with both metric filter and demonstration join", async () => {
    // Chat dataset: We filter on a float metric "haiku_rating" and join demonstration feedback.
    // In our fixtures, we expect that the intersection of rows having a "haiku_rating" above 0.8
    // and with demonstration feedback is 67.
    const chatParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_rating",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "id",
      },
      output_source: "demonstration",
    });
    const chatCount = await countRowsForDataset(chatParams);
    expect(chatCount).toBe(67);

    // JSON dataset: Similarly, we filter on a float metric "jaccard_similarity" and join demonstration feedback.
    // According to our fixtures, the expected intersection count is 0 as no elements have both a
    // "jaccard_similarity" above 0.8 and demonstration feedback.
    const jsonParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "jaccard_similarity",
        metric_type: "float",
        operator: ">",
        threshold: 0.8,
        join_on: "id",
      },
      output_source: "demonstration",
    });
    const jsonCount = await countRowsForDataset(jsonParams);
    expect(jsonCount).toBe(0);
  });
});
