import { expect, test } from "vitest";
import {
  countCommentFeedbackByTargetId,
  queryCommentFeedbackBoundsByTargetId,
  queryCommentFeedbackByTargetId,
  type CommentFeedbackRow,
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
});

test("countCommentFeedbackByTargetId", async () => {
  const count = await countCommentFeedbackByTargetId(
    "01942e26-4693-7e80-8591-47b98e25d721",
  );
  expect(count).toBe(26);
});

test("queryFeedbackBoundsByTargetId", async () => {
  const bounds = await queryCommentFeedbackBoundsByTargetId({
    target_id: "01942e26-4693-7e80-8591-47b98e25d721",
  });
  expect(bounds).toEqual({
    first_id: "01944339-00dc-7aa2-843a-74ebdf5d5d60",
    last_id: "01944339-0132-7fb2-b3bc-84537d686443",
  });
});
