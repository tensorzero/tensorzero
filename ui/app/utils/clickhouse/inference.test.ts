import { expect, test } from "vitest";
import {
  listInferencesWithPagination,
  countInferencesForVariant,
  countInferencesForFunction,
} from "./inference.server";

// Test countInferencesForFunction
test("countInferencesForFunction returns correct counts", async () => {
  const jsonCount = await countInferencesForFunction("extract_entities");
  expect(jsonCount).toBeGreaterThanOrEqual(604);

  const chatCount = await countInferencesForFunction("write_haiku");
  expect(chatCount).toBeGreaterThanOrEqual(804);
});

// Test countInferencesForVariant
test("countInferencesForVariant returns correct counts", async () => {
  const jsonCount = await countInferencesForVariant(
    "extract_entities",
    "gpt4o_initial_prompt",
  );
  expect(jsonCount).toBeGreaterThanOrEqual(132);

  const chatCount = await countInferencesForVariant(
    "write_haiku",
    "initial_prompt_gpt4o_mini",
  );
  expect(chatCount).toBeGreaterThanOrEqual(649);
});

// Tests for listInferencesWithPagination (new cursor-based pagination API)
test(
  "listInferencesWithPagination basic pagination",
  // TODO - lower this timeout when we figure out what's wrong with clickhouse lts
  { timeout: 20_000 },
  async () => {
    const result = await listInferencesWithPagination({
      limit: 10,
    });
    expect(result.inferences.length).toBe(10);
    expect(result.hasPreviousPage).toBe(false); // First page has no previous
    expect(result.hasNextPage).toBe(true); // Should have more pages

    // Verify IDs are in descending order
    for (let i = 1; i < result.inferences.length; i++) {
      expect(
        result.inferences[i - 1].inference_id >
          result.inferences[i].inference_id,
      ).toBe(true);
    }

    // Get second page using before
    const result2 = await listInferencesWithPagination({
      before: result.inferences[result.inferences.length - 1].inference_id,
      limit: 10,
    });
    expect(result2.inferences.length).toBe(10);
    expect(result2.hasPreviousPage).toBe(true); // We came from a newer page
  },
);

test(
  "listInferencesWithPagination pagination with before and after",
  { timeout: 10_000 },
  async () => {
    const LIMIT = 100;

    // First page (most recent)
    const firstResult = await listInferencesWithPagination({ limit: LIMIT });
    expect(firstResult.inferences.length).toBe(LIMIT);
    expect(firstResult.hasPreviousPage).toBe(false);
    expect(firstResult.hasNextPage).toBe(true);
    for (let i = 1; i < firstResult.inferences.length; i++) {
      expect(
        firstResult.inferences[i - 1].inference_id >
          firstResult.inferences[i].inference_id,
      ).toBe(true);
    }

    // Second page using before (going to older)
    const secondResult = await listInferencesWithPagination({
      before:
        firstResult.inferences[firstResult.inferences.length - 1].inference_id,
      limit: LIMIT,
    });
    expect(secondResult.inferences.length).toBe(LIMIT);
    expect(secondResult.hasPreviousPage).toBe(true);
    for (let i = 1; i < secondResult.inferences.length; i++) {
      expect(
        secondResult.inferences[i - 1].inference_id >
          secondResult.inferences[i].inference_id,
      ).toBe(true);
    }

    // Go back to newer using after
    const forwardResult = await listInferencesWithPagination({
      after:
        secondResult.inferences[secondResult.inferences.length - 1]
          .inference_id,
      limit: LIMIT,
    });
    expect(forwardResult.inferences.length).toBe(LIMIT);
    expect(forwardResult.hasNextPage).toBe(true); // We came from older page
    for (let i = 1; i < forwardResult.inferences.length; i++) {
      expect(
        forwardResult.inferences[i - 1].inference_id >
          forwardResult.inferences[i].inference_id,
      ).toBe(true);
    }
  },
);

test("listInferencesWithPagination after future timestamp is empty", async () => {
  const futureUUID = "ffffffff-ffff-7fff-ffff-ffffffffffff";
  const result = await listInferencesWithPagination({
    after: futureUUID,
    limit: 10,
  });
  expect(result.inferences.length).toBe(0);
  expect(result.hasPreviousPage).toBe(false);
  expect(result.hasNextPage).toBe(true); // We came from future, so there are older pages
});

test("listInferencesWithPagination before past timestamp is empty", async () => {
  const pastUUID = "00000000-0000-7000-0000-000000000000";
  const result = await listInferencesWithPagination({
    before: pastUUID,
    limit: 10,
  });
  expect(result.inferences.length).toBe(0);
  expect(result.hasNextPage).toBe(false); // No more older pages
  expect(result.hasPreviousPage).toBe(true); // We came from newer page
});

test("listInferencesWithPagination with function_name filter", async () => {
  const result = await listInferencesWithPagination({
    function_name: "extract_entities",
    limit: 10,
  });
  expect(result.inferences.length).toBe(10);

  // All inferences should be for the specified function
  for (const inference of result.inferences) {
    expect(inference.function_name).toBe("extract_entities");
  }

  // Verify IDs are in descending order
  for (let i = 1; i < result.inferences.length; i++) {
    expect(
      result.inferences[i - 1].inference_id > result.inferences[i].inference_id,
    ).toBe(true);
  }

  // Test pagination with before
  const result2 = await listInferencesWithPagination({
    function_name: "extract_entities",
    before: result.inferences[result.inferences.length - 1].inference_id,
    limit: 10,
  });
  expect(result2.inferences.length).toBe(10);
  for (const inference of result2.inferences) {
    expect(inference.function_name).toBe("extract_entities");
  }
});

test("listInferencesWithPagination with variant_name filter", async () => {
  const result = await listInferencesWithPagination({
    function_name: "extract_entities",
    variant_name: "gpt4o_initial_prompt",
    limit: 10,
  });
  expect(result.inferences.length).toBe(10);

  // All inferences should be for the specified function and variant
  for (const inference of result.inferences) {
    expect(inference.function_name).toBe("extract_entities");
    expect(inference.variant_name).toBe("gpt4o_initial_prompt");
  }

  // Verify IDs are in descending order
  for (let i = 1; i < result.inferences.length; i++) {
    expect(
      result.inferences[i - 1].inference_id > result.inferences[i].inference_id,
    ).toBe(true);
  }
});

test("listInferencesWithPagination with episode_id filter", async () => {
  const episodeId = "01942e26-618b-7b80-b492-34bed9f6d872";
  const LIMIT = 20;

  // First page
  const result = await listInferencesWithPagination({
    episode_id: episodeId,
    limit: LIMIT,
  });
  expect(result.inferences.length).toBe(LIMIT);
  expect(result.hasPreviousPage).toBe(false);
  expect(result.hasNextPage).toBe(true); // Episode has 35 inferences, so there's more

  // All inferences should be for the specified episode
  for (const inference of result.inferences) {
    expect(inference.episode_id).toBe(episodeId);
  }

  // Second page
  const result2 = await listInferencesWithPagination({
    episode_id: episodeId,
    before: result.inferences[result.inferences.length - 1].inference_id,
    limit: LIMIT,
  });
  // Should have remaining 15 inferences
  expect(result2.inferences.length).toBe(15);
  expect(result2.hasPreviousPage).toBe(true);
  expect(result2.hasNextPage).toBe(false); // Last page

  for (const inference of result2.inferences) {
    expect(inference.episode_id).toBe(episodeId);
  }
});
