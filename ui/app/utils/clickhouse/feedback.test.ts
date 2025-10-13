import { describe, expect, test, vi } from "vitest";
import { pollForFeedbackItem, queryMetricsWithFeedback } from "./feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";

describe("queryMetricsWithFeedback", () => {
  test("returns correct feedback counts for different metric types", async () => {
    // Test json function with multiple metric types
    const jsonResults = await queryMetricsWithFeedback({
      function_name: "extract_entities",
      inference_table: "JsonInference",
    });

    // Check boolean counts for JSON function
    expect(jsonResults.metrics).toContainEqual({
      function_name: "extract_entities",
      metric_name: "exact_match",
      metric_type: "boolean",
      feedback_count: 99,
    });

    // Check demonstration counts for JSON function
    expect(jsonResults.metrics).toContainEqual({
      function_name: "extract_entities",
      metric_name: "demonstration",
      metric_type: "demonstration",
      feedback_count: 100,
    });

    // Test chat function with float metrics
    const chatResults = await queryMetricsWithFeedback({
      function_name: "write_haiku",
      inference_table: "ChatInference",
    });

    expect(chatResults.metrics).toContainEqual({
      function_name: "write_haiku",
      metric_name: "haiku_rating",
      metric_type: "float",
      feedback_count: 491,
    });

    // Check demonstration counts for chat function
    expect(chatResults.metrics).toContainEqual({
      function_name: "write_haiku",
      metric_name: "demonstration",
      metric_type: "demonstration",
      feedback_count: 493,
    });
  });

  // Tests error handling for nonexistent functions
  test("returns empty array for nonexistent function", async () => {
    const results = await queryMetricsWithFeedback({
      function_name: "nonexistent_function",
      inference_table: "ChatInference",
    });

    expect(results.metrics).toEqual([]);
  });

  // Tests handling of metrics at different levels (inference vs episode)
  test("returns correct metrics for both inference and episode levels", async () => {
    const results = await queryMetricsWithFeedback({
      function_name: "write_haiku",
      inference_table: "ChatInference",
    });

    // Check inference level metric
    expect(results.metrics).toContainEqual({
      function_name: "write_haiku",
      metric_name: "haiku_rating",
      metric_type: "float",
      feedback_count: 491,
    });

    // Check episode level metric
    expect(results.metrics).toContainEqual({
      function_name: "write_haiku",
      metric_name: "haiku_rating_episode",
      metric_type: "float",
      feedback_count: 85,
    });
  });

  test("returns correct feedback counts for variant", async () => {
    const results = await queryMetricsWithFeedback({
      function_name: "write_haiku",
      variant_name: "initial_prompt_gpt4o_mini",
      inference_table: "ChatInference",
    });

    expect(results.metrics).toContainEqual({
      function_name: "write_haiku",
      metric_name: "haiku_rating",
      metric_type: "float",
      feedback_count: 491,
    });
  });
});

test("pollForFeedbackItem should find feedback when it exists", async () => {
  const targetId = "01942e26-4693-7e80-8591-47b98e25d721";
  const pageSize = 10;

  const dbClient = await getNativeDatabaseClient();
  // Run the queryFeedbackByTargetId function to return feedback with the target ID
  const originalQueryFeedback = await dbClient.queryFeedbackByTargetId({
    target_id: targetId,
    before: undefined,
    after: undefined,
    page_size: pageSize,
  });

  // Ensure we have feedback to test with
  expect(originalQueryFeedback.length).toBeGreaterThan(0);

  // Use the first feedback item's ID for our test
  const existingFeedbackId = originalQueryFeedback[0].id;

  // Poll for the existing feedback item
  const feedback = await pollForFeedbackItem(
    targetId,
    existingFeedbackId,
    pageSize,
    3, // Fewer retries for faster test
    50, // Shorter delay for faster test
  );

  // Verify the feedback was found
  expect(feedback.length).toBeGreaterThan(0);
  expect(feedback.some((f) => f.id === existingFeedbackId)).toBe(true);

  // Test with non-existent feedback ID
  const nonExistentId = "non-existent-feedback-id";
  const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

  const emptyFeedback = await pollForFeedbackItem(
    targetId,
    nonExistentId,
    pageSize,
    2, // Fewer retries for faster test
    50, // Shorter delay for faster test
  );

  // Verify warning was logged
  expect(consoleSpy).toHaveBeenCalledWith(
    `[TensorZero UI ${__APP_VERSION__}] Feedback ${nonExistentId} for target ${targetId} not found after 2 retries.`,
  );

  // Verify we still get feedback for the target, even though specific item wasn't found
  expect(emptyFeedback.length).toBeGreaterThan(0);
  expect(emptyFeedback.some((f) => f.id === nonExistentId)).toBe(false);

  consoleSpy.mockRestore();
});
