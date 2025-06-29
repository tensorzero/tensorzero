import { describe, expect, test } from "vitest";
import {
  countInferencesForEpisode,
  queryEpisodeTable,
  queryEpisodeTableBounds,
  queryInferenceById,
  queryInferenceTable,
  queryInferenceTableBounds,
  queryInferenceTableBoundsByEpisodeId,
  queryInferenceTableByEpisodeId,
  countInferencesByFunction,
  countInferencesForVariant,
  queryInferenceTableByVariantName,
  queryInferenceTableByFunctionName,
  queryInferenceTableBoundsByFunctionName,
  queryInferenceTableBoundsByVariantName,
  queryModelInferencesByInferenceId,
  getAdjacentInferenceIds,
  getAdjacentEpisodeIds,
} from "./inference.server";
import { countInferencesForFunction } from "./inference.server";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
  TextContent,
} from "./common";
import { clickhouseClient } from "./client.server";

// Test countInferencesForFunction
test("countInferencesForFunction returns correct counts", async () => {
  const jsonCount = await countInferencesForFunction("extract_entities", {
    type: "json",
    variants: {},
  });
  expect(jsonCount).toBe(604);

  const chatCount = await countInferencesForFunction("write_haiku", {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  });
  expect(chatCount).toBe(804);
});

// Test countInferencesForVariant
test("countInferencesForVariant returns correct counts", async () => {
  const jsonCount = await countInferencesForVariant(
    "extract_entities",
    { type: "json", variants: {} },
    "gpt4o_initial_prompt",
  );
  expect(jsonCount).toBe(132);

  const chatCount = await countInferencesForVariant(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "initial_prompt_gpt4o_mini",
  );
  expect(chatCount).toBe(649);
});

test("queryInferenceTable", async () => {
  const inferences = await queryInferenceTable({
    page_size: 10,
  });
  expect(inferences.length).toBe(10);

  // Verify IDs are in descending order
  for (let i = 1; i < inferences.length; i++) {
    expect(inferences[i - 1].id > inferences[i].id).toBe(true);
  }

  const inferences2 = await queryInferenceTable({
    before: inferences[inferences.length - 1].id,
    page_size: 10,
  });
  expect(inferences2.length).toBe(10);
});

test("queryInferenceTable pagination samples front and near-end pages correctly", async () => {
  const PAGE_SIZE = 100;

  // --- Front of the table (most recent entries) ---
  const firstPage = await queryInferenceTable({ page_size: PAGE_SIZE });
  expect(firstPage.length).toBe(PAGE_SIZE);
  for (let i = 1; i < firstPage.length; i++) {
    expect(firstPage[i - 1].id > firstPage[i].id).toBe(true);
  }

  const secondPage = await queryInferenceTable({
    before: firstPage[firstPage.length - 1].id,
    page_size: PAGE_SIZE,
  });
  expect(secondPage.length).toBe(PAGE_SIZE);
  for (let i = 1; i < secondPage.length; i++) {
    expect(secondPage[i - 1].id > secondPage[i].id).toBe(true);
  }

  // --- Near the end of the table (oldest entries) ---
  const bounds = await queryInferenceTableBounds();
  // bounds.last_id is the earliest (oldest) inference ID
  const lastID = bounds.last_id!;

  const endPage1 = await queryInferenceTable({
    before: lastID,
    page_size: PAGE_SIZE,
  });
  expect(endPage1.length).toBeGreaterThan(0);
  for (let i = 1; i < endPage1.length; i++) {
    expect(endPage1[i - 1].id > endPage1[i].id).toBe(true);
  }

  const endPage2 = await queryInferenceTable({
    before: endPage1[endPage1.length - 1].id,
    page_size: PAGE_SIZE,
  });
  // this may be empty if there are no more older entries
  expect(endPage2.length).toBeGreaterThanOrEqual(0);
  for (let i = 1; i < endPage2.length; i++) {
    expect(endPage2[i - 1].id > endPage2[i].id).toBe(true);
  }

  // Try to grab the last page by after
  const lastPageByAfter = await queryInferenceTable({
    after: endPage1[endPage1.length - 1].id,
    page_size: PAGE_SIZE,
  });
  expect(lastPageByAfter.length).toBe(PAGE_SIZE);
  for (let i = 1; i < lastPageByAfter.length; i++) {
    expect(lastPageByAfter[i - 1].id > lastPageByAfter[i].id).toBe(true);
  }
});

test("queryInferenceTable pages through results correctly using after with inference ID", async () => {
  const PAGE_SIZE = 20;
  // Get a limited sample of inferences instead of the full table
  const sampleSize = 100;

  // Now, page forward using after through our sample
  let results: Awaited<ReturnType<typeof queryInferenceTable>> = [];
  let after: string | undefined = undefined;
  let pageCount = 0;

  // Only page through our sample size
  while (results.length < sampleSize) {
    const page = await queryInferenceTable({
      page_size: PAGE_SIZE,
      after,
    });
    if (page.length === 0) break;

    // IDs should be in descending order within the page
    for (let i = 1; i < page.length; i++) {
      expect(page[i - 1].id > page[i].id).toBe(true);
    }

    results = results.concat(page);
    after = page[page.length - 1].id;
    pageCount++;

    if (page.length < PAGE_SIZE) break;

    // Safety check to avoid infinite loops
    if (pageCount > 10) break;
  }

  // Should have paged at least once
  expect(pageCount).toBeGreaterThanOrEqual(1);
});

test("queryInferenceTable after future timestamp is empty", async () => {
  // Create a future timestamp UUID - this will be larger than any existing ID
  const futureUUID = "ffffffff-ffff-7fff-ffff-ffffffffffff";
  const inferences = await queryInferenceTable({
    after: futureUUID,
    page_size: 10,
  });
  expect(inferences.length).toBe(0);
});

test("queryInferenceTable before past timestamp is empty", async () => {
  // Create a past timestamp UUID - this will be smaller than any existing ID
  const pastUUID = "00000000-0000-7000-0000-000000000000";
  const inferences = await queryInferenceTable({
    before: pastUUID,
    page_size: 10,
  });
  expect(inferences.length).toBe(0);
});

test("queryInferenceTableByEpisodeId pages through initial and final pages correctly using before with episode_id", async () => {
  const PAGE_SIZE = 20;
  const episodeId = "01942e26-618b-7b80-b492-34bed9f6d872";

  // First page
  const firstPage = await queryInferenceTableByEpisodeId({
    page_size: PAGE_SIZE,
    episode_id: episodeId,
  });
  // Should be exactly PAGE_SIZE
  expect(firstPage.length).toBe(PAGE_SIZE);
  // IDs in descending order
  for (let i = 1; i < firstPage.length; i++) {
    expect(firstPage[i - 1].id > firstPage[i].id).toBe(true);
  }

  // Next (and in this case final) page
  const lastPage = await queryInferenceTableByEpisodeId({
    before: firstPage[firstPage.length - 1].id,
    page_size: PAGE_SIZE,
    episode_id: episodeId,
  });
  // Should be smaller or equal to PAGE_SIZE (here 15)
  expect(lastPage.length).toBeLessThanOrEqual(PAGE_SIZE);
  // IDs in descending order
  for (let i = 1; i < lastPage.length; i++) {
    expect(lastPage[i - 1].id > lastPage[i].id).toBe(true);
  }

  // Verify total elements matches expected count
  expect(firstPage.length + lastPage.length).toBe(35);
});

test("queryInferenceTableByEpisodeId pages through a sample of results correctly using after with episode_id", async () => {
  const PAGE_SIZE = 20;
  const episodeId = "01942e26-469a-7553-af00-cb0495dc7bb5";

  // Get the first page
  const firstPage = await queryInferenceTableByEpisodeId({
    page_size: PAGE_SIZE,
    episode_id: episodeId,
  });

  // Verify first page properties
  expect(firstPage.length).toBe(PAGE_SIZE);
  for (let i = 1; i < firstPage.length; i++) {
    expect(firstPage[i - 1].id > firstPage[i].id).toBe(true);
  }

  // Get the second page using the last ID of the first page
  const secondPage = await queryInferenceTableByEpisodeId({
    before: firstPage[firstPage.length - 1].id,
    page_size: PAGE_SIZE,
    episode_id: episodeId,
  });

  // Verify second page properties
  expect(secondPage.length).toBe(PAGE_SIZE);
  for (let i = 1; i < secondPage.length; i++) {
    expect(secondPage[i - 1].id > secondPage[i].id).toBe(true);
  }

  // Now test paging forward using after
  // Get a page starting after the first item of the second page
  const forwardPage = await queryInferenceTableByEpisodeId({
    after: secondPage[0].id,
    page_size: PAGE_SIZE,
    episode_id: episodeId,
  });

  // Verify forward paging works
  expect(forwardPage.length).toBeGreaterThan(0);
  expect(forwardPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify IDs are in descending order
  for (let i = 1; i < forwardPage.length; i++) {
    expect(forwardPage[i - 1].id > forwardPage[i].id).toBe(true);
  }

  // Get the last page to verify we can reach the end
  let lastPage = firstPage;
  let nextPage = secondPage;

  // Just get a couple more pages to avoid too many queries
  for (let i = 0; i < 2 && nextPage.length === PAGE_SIZE; i++) {
    lastPage = nextPage;
    nextPage = await queryInferenceTableByEpisodeId({
      before: lastPage[lastPage.length - 1].id,
      page_size: PAGE_SIZE,
      episode_id: episodeId,
    });
  }

  // Verify we can get to the end if needed
  if (nextPage.length < PAGE_SIZE) {
    // We reached the last page
    expect(nextPage.length).toBeLessThan(PAGE_SIZE);
  }

  // Verify total count is as expected (42 from previous test)
  const totalCount = await countInferencesForEpisode(episodeId);
  expect(totalCount).toBe(43);
});

// queryInferenceTableBounds and queryEpisodeTableBounds are the same because the early inferences are in singleton episodes.
test("queryInferenceTableBounds", async () => {
  const bounds = await queryInferenceTableBounds();
  expect(bounds.first_id).toBe("01934c9a-be70-74e2-8e6d-8eb19531638c");
  expect(bounds.last_id).toBe("0197177a-7c00-70a2-82a6-741f60a03b2e");
});

test("queryEpisodeTableBounds", async () => {
  const bounds = await queryEpisodeTableBounds();
  expect(bounds.first_id).toBe("01934c9a-be70-74e2-8e6d-8eb19531638c");
  expect(bounds.last_id).toBe("0197177a-7c00-70a2-82a6-741f60a03b2e");
});

test("queryInferenceTableBounds with episode_id", async () => {
  const bounds = await queryInferenceTableBoundsByEpisodeId({
    episode_id: "01942e26-6497-7910-89d6-d9d1c735d3df",
  });
  expect(bounds.first_id).toBe("01942e26-6e50-7fa0-8d61-9fd730a73a8b");
  expect(bounds.last_id).toBe("01942e27-5a2b-75b3-830c-094f82096270");
});

test("queryInferenceTableBounds with invalid episode_id", async () => {
  const bounds = await queryInferenceTableBoundsByEpisodeId({
    episode_id: "01942e26-6497-7910-89d6-d9c1c735d3df",
  });
  expect(bounds.first_id).toBe(null);
  expect(bounds.last_id).toBe(null);
});

test("queryInferenceTableByFunctionName", async () => {
  const inferences = await queryInferenceTableByFunctionName({
    function_name: "extract_entities",
    page_size: 10,
  });
  expect(inferences.length).toBe(10);

  // Verify IDs are in descending order
  for (let i = 1; i < inferences.length; i++) {
    expect(inferences[i - 1].id > inferences[i].id).toBe(true);
  }

  // Test pagination with before
  const inferences2 = await queryInferenceTableByFunctionName({
    function_name: "extract_entities",
    before: inferences[inferences.length - 1].id,
    page_size: 10,
  });
  expect(inferences2.length).toBe(10);

  // Test pagination with after
  const inferences3 = await queryInferenceTableByFunctionName({
    function_name: "extract_entities",
    after: inferences[0].id,
    page_size: 10,
  });
  expect(inferences3.length).toBe(0);
});

test("queryInferenceTableBoundsByFunctionName", async () => {
  const bounds = await queryInferenceTableBoundsByFunctionName({
    function_name: "extract_entities",
  });
  expect(bounds.first_id).toBe("01934c9a-be70-74e2-8e6d-8eb19531638c");
  expect(bounds.last_id).toBe("0196374c-2c92-74b3-843f-ffa611b577b4");
});

test("queryInferenceTableByVariantName", async () => {
  const inferences = await queryInferenceTableByVariantName({
    function_name: "extract_entities",
    variant_name: "gpt4o_initial_prompt",
    page_size: 10,
  });
  expect(inferences.length).toBe(10);

  // Verify IDs are in descending order
  for (let i = 1; i < inferences.length; i++) {
    expect(inferences[i - 1].id > inferences[i].id).toBe(true);
  }

  // Test pagination with before
  const inferences2 = await queryInferenceTableByVariantName({
    function_name: "extract_entities",
    variant_name: "gpt4o_initial_prompt",
    before: inferences[inferences.length - 1].id,
    page_size: 10,
  });
  expect(inferences2.length).toBe(10);

  // Test pagination with after
  const inferences3 = await queryInferenceTableByVariantName({
    function_name: "extract_entities",
    variant_name: "gpt4o_initial_prompt",
    after: inferences[0].id,
    page_size: 10,
  });
  expect(inferences3.length).toBe(0);
});

test("queryInferenceTableBoundsByVariantName", async () => {
  const bounds = await queryInferenceTableBoundsByVariantName({
    function_name: "extract_entities",
    variant_name: "gpt4o_initial_prompt",
  });
  expect(bounds.first_id).toBe("01939adf-0f50-79d0-8d55-7a009fcc5e32");
  expect(bounds.last_id).toBe("0196368e-5505-7721-88d2-654cd26483b4");
});

test(
  "queryEpisodeTable",
  // https://tensorzero.slack.com/archives/C06FDMR1YKF/p1747844085031149?thread_ts=1747793217.140669&cid=C06FDMR1YKF
  { timeout: 10_000 },
  async () => {
    const episodes = await queryEpisodeTable({
      page_size: 10,
    });
    expect(episodes.length).toBe(10);

    // Verify last_inference_ids are in descending order
    for (let i = 1; i < episodes.length; i++) {
      expect(
        episodes[i - 1].last_inference_id > episodes[i].last_inference_id,
      ).toBe(true);
    }

    // Test pagination with before
    const episodes2 = await queryEpisodeTable({
      before: episodes[episodes.length - 1].last_inference_id,
      page_size: 10,
    });
    expect(episodes2.length).toBe(10);

    // Test pagination with after on the last inference id
    const episodes3 = await queryEpisodeTable({
      after: episodes[0].last_inference_id,
      page_size: 10,
    });
    expect(episodes3.length).toBe(0);

    // Test that before and after together throws error
    await expect(
      queryEpisodeTable({
        before: episodes[0].last_inference_id,
        after: episodes[0].last_inference_id,
        page_size: 10,
      }),
    ).rejects.toThrow("Cannot specify both 'before' and 'after' parameters");

    // Verify each episode has valid data
    for (const episode of episodes) {
      expect(typeof episode.episode_id).toBe("string");
      expect(episode.count).toBeGreaterThan(0);
      expect(episode.start_time).toBeDefined();
      expect(episode.end_time).toBeDefined();
      expect(typeof episode.last_inference_id).toBe("string");
      // Start time should be before or equal to end time
      expect(new Date(episode.start_time) <= new Date(episode.end_time)).toBe(
        true,
      );
    }
  },
);

test("countInferencesForEpisode", async () => {
  const count = await countInferencesForEpisode(
    "01942e26-549f-7153-ac56-dd1d23d30f8c",
  );
  expect(count).toBe(43);
});

test("countInferencesForEpisode with invalid episode_id", async () => {
  const count = await countInferencesForEpisode(
    "01942e26-549f-7153-ac56-dd1d23d30f8d",
  );
  expect(count).toBe(0);
});

test("queryInferenceById for chat inference", async () => {
  const inference = await queryInferenceById(
    "01942e26-910b-7ab1-a645-46bc4463a001",
  );
  expect(inference?.function_type).toBe("chat");
  expect(inference?.input.messages.length).toBeGreaterThan(0);
  const output = inference?.output as ContentBlockOutput[];
  const firstOutput = output[0] as TextContent;
  expect(firstOutput.type).toBe("text");
  expect(firstOutput.text).toBe("Yes.");
});

test("queryInferenceById for missing inference", async () => {
  const inference = await queryInferenceById(
    "01942e26-910b-7ab1-a645-46bc4463a000",
  );
  expect(inference).toBeNull();
});

test("queryInferenceById for json inference", async () => {
  const inference = await queryInferenceById(
    "01942e26-88ab-7331-8293-de75cc2b88a7",
  );
  expect(inference?.function_type).toBe("json");
  expect(inference?.input.messages.length).toBe(0);
  const output = inference?.output as JsonInferenceOutput;
  expect(output.parsed).toBeDefined();
});
test("countInferencesByFunction", async () => {
  const countsInfo = await countInferencesByFunction();
  expect(countsInfo).toEqual(
    expect.arrayContaining([
      {
        count: 204,
        function_name: "tensorzero::llm_judge::entity_extraction::count_sports",
        max_timestamp: "2025-04-15T02:34:22Z",
      },
      {
        count: 604,
        function_name: "extract_entities",
        max_timestamp: "2025-04-15T02:34:21Z",
      },
      {
        count: 310,
        function_name: "tensorzero::llm_judge::haiku::topic_starts_with_f",
        max_timestamp: "2025-04-15T02:33:10Z",
      },
      {
        count: 804,
        function_name: "write_haiku",
        max_timestamp: "2025-05-12T21:59:20Z",
      },
      {
        count: 2,
        function_name: "tensorzero::default",
        max_timestamp: "2025-05-23T15:49:52Z",
      },
      {
        count: 1,
        function_name: "foo",
        max_timestamp: "2025-02-13T22:29:20Z",
      },
    ]),
  );
});

test("queryModelInferencesByInferenceId", async () => {
  const modelInferences = await queryModelInferencesByInferenceId(
    "0195aef6-6cee-75e3-9097-f7bdf6e9c9af",
  );
  expect(modelInferences.length).toBe(1);
  const firstInference = modelInferences[0];
  expect(firstInference.id).toBe("0195aef6-77a9-7ce1-8910-016c2bef9cec");
  expect(firstInference.input_messages.length).toBe(1);
  expect(firstInference.output.length).toBe(1);
  expect(firstInference.output[0].type).toBe("text");
  expect(!firstInference.cached);
});

describe("getAdjacentInferenceIds", () => {
  test("returns adjacent inference ids", async () => {
    const adjacentInferenceIds = await getAdjacentInferenceIds(
      "01942e26-910b-7ab1-a645-46bc4463a001",
    );
    expect(adjacentInferenceIds.previous_id).toBe(
      "01942e26-9026-76e0-bf84-27038739ec33",
    );
    expect(adjacentInferenceIds.next_id).toBe(
      "01942e26-9128-71d2-bed6-aee96bb3e181",
    );
  });

  test("returns null for previous inference id if current inference is first", async () => {
    const resultSet = await clickhouseClient.query({
      query:
        "SELECT uint_to_uuid(min(id_uint)) as first_inference_id FROM InferenceById",
      format: "JSON",
    });
    const firstInferenceId = await resultSet.json<{
      first_inference_id: string;
    }>();
    const adjacentInferenceIds = await getAdjacentInferenceIds(
      firstInferenceId.data[0].first_inference_id,
    );
    expect(adjacentInferenceIds.previous_id).toBeNull();
    expect(adjacentInferenceIds.next_id).toBe(
      "01934c9a-be70-7d72-a722-744cb572eb49",
    );
  });

  test("returns null for next inference id if current inference is last", async () => {
    const resultSet = await clickhouseClient.query({
      query:
        "SELECT uint_to_uuid(max(id_uint)) as last_inference_id FROM InferenceById",
      format: "JSON",
    });
    const lastInferenceId = await resultSet.json<{
      last_inference_id: string;
    }>();
    const adjacentInferenceIds = await getAdjacentInferenceIds(
      lastInferenceId.data[0].last_inference_id,
    );
    expect(adjacentInferenceIds.previous_id).toBe(
      "0197177a-7c00-70a2-82a6-72ac87d2ff77",
    );
    expect(adjacentInferenceIds.next_id).toBeNull();
  });
});

describe("getAdjacentEpisodeIds", () => {
  test("returns adjacent episode ids", async () => {
    const adjacentEpisodeIds = await getAdjacentEpisodeIds(
      "01942e26-549f-7153-ac56-dd1d23d30f8c",
    );
    expect(adjacentEpisodeIds.previous_id).toBe(
      "01942e26-5392-7652-ad59-734198888520",
    );
    expect(adjacentEpisodeIds.next_id).toBe(
      "01942e26-54a2-71d1-ad80-3629b6cb18a3",
    );
  });

  test("returns null for previous episode id if current episode is first", async () => {
    const resultSet = await clickhouseClient.query({
      query:
        "SELECT uint_to_uuid(min(episode_id_uint)) as first_episode_id FROM InferenceByEpisodeId",
      format: "JSON",
    });
    const firstEpisodeId = await resultSet.json<{
      first_episode_id: string;
    }>();

    const adjacentEpisodeIds = await getAdjacentEpisodeIds(
      firstEpisodeId.data[0].first_episode_id,
    );
    expect(adjacentEpisodeIds.previous_id).toBeNull();
    expect(adjacentEpisodeIds.next_id).toBe(
      "0192ced0-9486-7491-9b60-42dd2ef9194e",
    );
  });

  test("returns null for next episode id if current episode is last", async () => {
    const resultSet = await clickhouseClient.query({
      query:
        "SELECT uint_to_uuid(max(episode_id_uint)) as last_episode_id FROM InferenceByEpisodeId",
      format: "JSON",
    });
    const lastEpisodeId = await resultSet.json<{
      last_episode_id: string;
    }>();

    const adjacentEpisodeIds = await getAdjacentEpisodeIds(
      lastEpisodeId.data[0].last_episode_id,
    );
    expect(adjacentEpisodeIds.previous_id).toBe(
      "0aaeef58-3633-7f27-9393-65bd98491026",
    );
    expect(adjacentEpisodeIds.next_id).toBeNull();
  });
});
