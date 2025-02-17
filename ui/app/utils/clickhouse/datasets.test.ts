import { countRowsForDataset, DatasetQueryParamsSchema } from "./datasets";
import { expect, test, describe } from "vitest";

describe("countRowsForDataset", () => {
  test("returns the correct number of rows for a specific function", async () => {
    const dataset_params = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
    });
    const rows = await countRowsForDataset(dataset_params);
    expect(rows).toBe(494);
  });

  test("returns the correct number of rows for a specific variant", async () => {
    const dataset_params = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      variant_name: "llama_8b_initial_prompt",
    });
    const rows = await countRowsForDataset(dataset_params);
    expect(rows).toBe(110);
  });

  test("throws an error if function_name is not provided but variant_name is", async () => {
    await expect(async () => {
      const dataset_params = DatasetQueryParamsSchema.parse({
        inferenceType: "chat",
        variant_name: "test",
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
        join_on: "inference_id",
      },
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
        join_on: "inference_id",
      },
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
        join_on: "inference_id",
      },
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
        join_on: "inference_id",
      },
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
        join_on: "inference_id",
      },
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
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(11);
  });

  test("returns correct count for demonstration metrics", async () => {
    const jsonDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "json",
      function_name: "extract_entities",
      join_demonstrations: true,
    });
    const jsonRows = await countRowsForDataset(jsonDatasetParams);
    expect(jsonRows).toBe(100);

    const chatDatasetParams = DatasetQueryParamsSchema.parse({
      inferenceType: "chat",
      function_name: "write_haiku",
      join_demonstrations: true,
    });
    const chatRows = await countRowsForDataset(chatDatasetParams);
    expect(chatRows).toBe(493);
  });
});
