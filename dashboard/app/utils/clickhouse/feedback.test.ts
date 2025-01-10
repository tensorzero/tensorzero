import { expect, test } from "vitest";
import {
  countCommentFeedbackByTargetId,
  queryCommentFeedbackBoundsByTargetId,
  queryCommentFeedbackByTargetId,
  queryFeedbackByTargetId,
  type BooleanMetricFeedbackRow,
  type CommentFeedbackRow,
  type FeedbackRow,
  type FloatMetricFeedbackRow,
} from "./feedback";

test("queryCommentFeedbackByTargetId", async () => {
  const target_id = "01942e26-4693-7e80-8591-47b98e25d721";
  const page_size = 10;

  // Get first page
  const firstPage = await queryCommentFeedbackByTargetId({
    target_id,
    page_size,
  });
  expect(firstPage).toHaveLength(10);

  // Get second page
  const secondPage = await queryCommentFeedbackByTargetId({
    target_id,
    before: firstPage[firstPage.length - 1].id,
    page_size,
  });
  expect(secondPage).toHaveLength(10);

  // Get third page
  const thirdPage = await queryCommentFeedbackByTargetId({
    target_id,
    before: secondPage[secondPage.length - 1].id,
    page_size,
  });
  expect(thirdPage).toHaveLength(6);

  // Check that all pages are sorted by id in descending order
  const checkSorting = (feedback: CommentFeedbackRow[]) => {
    for (let i = 1; i < feedback.length; i++) {
      expect(feedback[i - 1].id > feedback[i].id).toBe(true);
    }
  };

  checkSorting(firstPage);
  checkSorting(secondPage);
  checkSorting(thirdPage);

  // Check total number of items
  expect(firstPage.length + secondPage.length + thirdPage.length).toBe(26);

  const emptyFeedback = await queryCommentFeedbackByTargetId({
    target_id: "01942e26-4693-7e80-8591-47b98e25d711",
    page_size: 10,
  });
  expect(emptyFeedback).toHaveLength(0);

  // Test paging backwards from a specific ID
  // There will be 1 less item in the first page because the last item is excluded
  const startId = "01944339-0132-7fb2-b3bc-84537d686443"; // Last ID
  const firstBackwardPage = await queryCommentFeedbackByTargetId({
    target_id,
    before: startId,
    page_size,
  });
  expect(firstBackwardPage).toHaveLength(10);

  const secondBackwardPage = await queryCommentFeedbackByTargetId({
    target_id,
    before: firstBackwardPage[firstBackwardPage.length - 1].id,
    page_size,
  });
  expect(secondBackwardPage).toHaveLength(10);

  const thirdBackwardPage = await queryCommentFeedbackByTargetId({
    target_id,
    before: secondBackwardPage[secondBackwardPage.length - 1].id,
    page_size,
  });
  expect(thirdBackwardPage).toHaveLength(5);

  // Check that backward pages are sorted by id in descending order
  checkSorting(firstBackwardPage);
  checkSorting(secondBackwardPage);
  checkSorting(thirdBackwardPage);

  // Check total number of items matches
  expect(
    firstBackwardPage.length +
      secondBackwardPage.length +
      thirdBackwardPage.length,
  ).toBe(25);
});

test("countCommentFeedbackByTargetId", async () => {
  const count = await countCommentFeedbackByTargetId(
    "01942e26-4693-7e80-8591-47b98e25d721",
  );
  expect(count).toBe(26);

  const emptyCount = await countCommentFeedbackByTargetId(
    "01942e26-4693-7e80-8591-47b98e25d711",
  );
  expect(emptyCount).toBe(0);
});

test("queryFeedbackBoundsByTargetId", async () => {
  const bounds = await queryCommentFeedbackBoundsByTargetId({
    target_id: "01942e26-4693-7e80-8591-47b98e25d721",
  });
  expect(bounds).toEqual({
    first_id: "01944339-00dc-7aa2-843a-74ebdf5d5d60",
    last_id: "01944339-0132-7fb2-b3bc-84537d686443",
  });

  const emptyBounds = await queryCommentFeedbackBoundsByTargetId({
    target_id: "01942e26-4693-7e80-8591-47b98e25d711",
  });
  expect(emptyBounds).toEqual({
    first_id: undefined,
    last_id: undefined,
  });
});

test("queryFeedbackByTargetId", async () => {
  const target_id = "01942e26-4693-7e80-8591-47b98e25d721";
  const page_size = 10;

  // Get first page
  const firstPage = await queryFeedbackByTargetId({
    target_id,
    page_size,
  });
  expect(firstPage).toHaveLength(10);

  // Get second page
  const secondPage = await queryFeedbackByTargetId({
    target_id,
    before: firstPage[firstPage.length - 1].id,
    page_size,
  });
  expect(secondPage).toHaveLength(10);

  // Get third page
  const thirdPage = await queryFeedbackByTargetId({
    target_id,
    before: secondPage[secondPage.length - 1].id,
    page_size,
  });
  expect(thirdPage).toHaveLength(8);

  // Check that all pages are sorted by id in descending order
  const checkSorting = (feedback: FeedbackRow[]) => {
    for (let i = 1; i < feedback.length; i++) {
      expect(feedback[i - 1].id > feedback[i].id).toBe(true);
    }
  };

  checkSorting(firstPage);
  checkSorting(secondPage);
  checkSorting(thirdPage);

  // The last element should be a float metric
  const lastElement = thirdPage[thirdPage.length - 1] as FloatMetricFeedbackRow;
  expect(lastElement.metric_name).toBe("num_questions");
  expect(lastElement.value).toBe(5);

  // The first element should be a boolean metric
  const firstElement = firstPage[0] as BooleanMetricFeedbackRow;
  expect(firstElement.metric_name).toBe("solved");
  expect(firstElement.value).toBe(true);

  // Check total number of items
  expect(firstPage.length + secondPage.length + thirdPage.length).toBe(28);

  const emptyFeedback = await queryFeedbackByTargetId({
    target_id: "01942e26-4693-7e80-8591-47b98e25d711",
    page_size: 10,
  });
  expect(emptyFeedback).toHaveLength(0);

  // Test paging backwards from a specific ID
  const startId = firstPage[0].id;
  const firstBackwardPage = await queryFeedbackByTargetId({
    target_id,
    before: startId,
    page_size,
  });
  expect(firstBackwardPage).toHaveLength(10);

  const secondBackwardPage = await queryFeedbackByTargetId({
    target_id,
    before: firstBackwardPage[firstBackwardPage.length - 1].id,
    page_size,
  });
  expect(secondBackwardPage).toHaveLength(10);

  const thirdBackwardPage = await queryFeedbackByTargetId({
    target_id,
    before: secondBackwardPage[secondBackwardPage.length - 1].id,
    page_size,
  });
  expect(thirdBackwardPage).toHaveLength(7);

  // Check that backward pages are sorted by id in descending order
  checkSorting(firstBackwardPage);
  checkSorting(secondBackwardPage);
  checkSorting(thirdBackwardPage);

  // Check total number of items matches
  expect(
    firstBackwardPage.length +
      secondBackwardPage.length +
      thirdBackwardPage.length,
  ).toBe(27);
});
