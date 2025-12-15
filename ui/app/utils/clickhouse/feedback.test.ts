import { expect, test, vi } from "vitest";
import { pollForFeedbackItem } from "./feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";

test("pollForFeedbackItem should find feedback when it exists", async () => {
  const targetId = "01942e26-4693-7e80-8591-47b98e25d721";
  const limit = 10;

  const dbClient = await getNativeDatabaseClient();
  // Run the queryFeedbackByTargetId function to return feedback with the target ID
  const originalQueryFeedback = await dbClient.queryFeedbackByTargetId({
    target_id: targetId,
    limit,
  });

  // Ensure we have feedback to test with
  expect(originalQueryFeedback.length).toBeGreaterThan(0);

  // Use the first feedback item's ID for our test
  const existingFeedbackId = originalQueryFeedback[0].id;

  // Poll for the existing feedback item
  const feedback = await pollForFeedbackItem(
    targetId,
    existingFeedbackId,
    limit,
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
    limit,
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
