import { expect, test } from "vitest";
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
} from "./inference";
import { countInferencesForFunction } from "./inference";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
  TextContent,
} from "./common";

// Test countInferencesForFunction
test("countInferencesForFunction returns correct counts", async () => {
  const jsonCount = await countInferencesForFunction(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
  );
  expect(jsonCount).toBe(400);

  const chatCount = await countInferencesForFunction(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
  );
  expect(chatCount).toBe(494);
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

test("queryInferenceTable pages through all results correctly using before", async () => {
  const PAGE_SIZE = 100;
  let currentPage = await queryInferenceTable({
    page_size: PAGE_SIZE,
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);
    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(currentPage[i - 1].id > currentPage[i].id).toBe(true);
    }

    // Get next page using last item's ID as cursor
    currentPage = await queryInferenceTable({
      before: currentPage[currentPage.length - 1].id,
      page_size: PAGE_SIZE,
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements
  expect(totalElements).toBe(2475);

  // We should have seen at least one full page
  expect(numFullPages).toBeGreaterThan(0);
});

test("queryInferenceTable pages through all results correctly using after", async () => {
  const PAGE_SIZE = 100;

  // First get to the last page to find the earliest ID
  let currentPage = await queryInferenceTable({
    page_size: PAGE_SIZE,
  });

  while (currentPage.length === PAGE_SIZE) {
    currentPage = await queryInferenceTable({
      before: currentPage[currentPage.length - 1].id,
      page_size: PAGE_SIZE,
    });
  }

  // Now we have the earliest ID, let's page forward using after
  const firstId = currentPage[currentPage.length - 1].id;
  currentPage = await queryInferenceTable({
    after: firstId,
    page_size: PAGE_SIZE,
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);

    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(currentPage[i - 1].id > currentPage[i].id).toBe(true);
    }

    // Get next page using first item's ID as cursor
    currentPage = await queryInferenceTable({
      after: currentPage[0].id,
      page_size: PAGE_SIZE,
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements matches the previous test
  expect(totalElements).toBe(2474); // One less than with before because we excluded the first ID

  // We should have seen at least one full page
  expect(numFullPages).toBeGreaterThan(0);
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

test("queryInferenceTableByEpisodeId pages through all results correctly using before with episode_id", async () => {
  const PAGE_SIZE = 20;
  let currentPage = await queryInferenceTableByEpisodeId({
    page_size: PAGE_SIZE,
    episode_id: "01942e26-618b-7b80-b492-34bed9f6d872",
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);
    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(currentPage[i - 1].id > currentPage[i].id).toBe(true);
    }

    // Get next page using last item's ID as cursor
    currentPage = await queryInferenceTableByEpisodeId({
      before: currentPage[currentPage.length - 1].id,
      page_size: PAGE_SIZE,
      episode_id: "01942e26-618b-7b80-b492-34bed9f6d872",
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements
  expect(totalElements).toBe(35);

  // We should have seen at least one full page
  expect(numFullPages).toBeGreaterThan(0);
});

test("queryInferenceTableByEpisodeId pages through all results correctly using after with episode_id", async () => {
  const PAGE_SIZE = 20;

  // First get to the last page to find the earliest ID
  let currentPage = await queryInferenceTableByEpisodeId({
    page_size: PAGE_SIZE,
    episode_id: "01942e26-469a-7553-af00-cb0495dc7bb5",
  });

  while (currentPage.length === PAGE_SIZE) {
    currentPage = await queryInferenceTableByEpisodeId({
      before: currentPage[currentPage.length - 1].id,
      page_size: PAGE_SIZE,
      episode_id: "01942e26-469a-7553-af00-cb0495dc7bb5",
    });
  }

  // Now we have the earliest ID, let's page forward using after
  const firstId = currentPage[currentPage.length - 1].id;
  currentPage = await queryInferenceTableByEpisodeId({
    after: firstId,
    page_size: PAGE_SIZE,
    episode_id: "01942e26-469a-7553-af00-cb0495dc7bb5",
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);

    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(currentPage[i - 1].id > currentPage[i].id).toBe(true);
    }

    // Get next page using first item's ID as cursor
    currentPage = await queryInferenceTableByEpisodeId({
      after: currentPage[0].id,
      page_size: PAGE_SIZE,
      episode_id: "01942e26-469a-7553-af00-cb0495dc7bb5",
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements matches the previous test
  expect(totalElements).toBe(42);

  // We should have seen at least two full pages
  expect(numFullPages).toBeGreaterThan(1);
});

// queryInferenceTableBounds and queryEpisodeTableBounds are the same because the early inferences are in singleton episodes.
test("queryInferenceTableBounds", async () => {
  const bounds = await queryInferenceTableBounds();
  expect(bounds.first_id).toBe("0192ced0-9873-70e2-ade5-dc5b8faea232");
  expect(bounds.last_id).toBe("01942e28-4a3c-7873-b94d-402a9cc83f2a");
});

test("queryEpisodeTableBounds", async () => {
  const bounds = await queryEpisodeTableBounds();
  expect(bounds.first_id).toBe("0192ced0-9873-70e2-ade5-dc5b8faea232");
  expect(bounds.last_id).toBe("01942e28-4a3c-7873-b94d-402a9cc83f2a");
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

test("queryEpisodeTable", async () => {
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
});

test("queryEpisodeTable pages through all results correctly using before", async () => {
  const PAGE_SIZE = 100;
  let currentPage = await queryEpisodeTable({
    page_size: PAGE_SIZE,
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);
    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(
        currentPage[i - 1].last_inference_id > currentPage[i].last_inference_id,
      ).toBe(true);
    }

    // Get next page using last item's ID as cursor
    currentPage = await queryEpisodeTable({
      before: currentPage[currentPage.length - 1].last_inference_id,
      page_size: PAGE_SIZE,
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements
  expect(totalElements).toBe(944);

  // We should have seen at least 9 full pages
  expect(numFullPages).toBeGreaterThan(8);
});

test("queryEpisodeTable pages through all results correctly using after", async () => {
  const PAGE_SIZE = 100;

  // First get to the last page to find the earliest ID
  let currentPage = await queryEpisodeTable({
    page_size: PAGE_SIZE,
  });

  while (currentPage.length === PAGE_SIZE) {
    currentPage = await queryEpisodeTable({
      before: currentPage[currentPage.length - 1].last_inference_id,
      page_size: PAGE_SIZE,
    });
  }

  // Now we have the earliest ID, let's page forward using after
  const firstId = currentPage[currentPage.length - 1].last_inference_id;
  currentPage = await queryEpisodeTable({
    after: firstId,
    page_size: PAGE_SIZE,
  });

  // Keep track of how many full pages we've seen
  let numFullPages = 0;
  let totalElements = 0;

  while (currentPage.length === PAGE_SIZE) {
    totalElements += currentPage.length;

    // Verify each page is the correct size
    expect(currentPage.length).toBe(PAGE_SIZE);

    // Verify IDs are in descending order within each page
    for (let i = 1; i < currentPage.length; i++) {
      expect(
        currentPage[i - 1].last_inference_id > currentPage[i].last_inference_id,
      ).toBe(true);
    }

    // Get next page using first item's ID as cursor
    currentPage = await queryEpisodeTable({
      after: currentPage[0].last_inference_id,
      page_size: PAGE_SIZE,
    });

    numFullPages++;
  }

  // Add the remaining elements from the last page
  totalElements += currentPage.length;

  // The last page should have fewer items than PAGE_SIZE
  // (unless the total happens to be exactly divisible by PAGE_SIZE)
  expect(currentPage.length).toBeLessThanOrEqual(PAGE_SIZE);

  // Verify total number of elements matches the previous test
  expect(totalElements).toBe(943); // One less than with before because we excluded the first ID

  // We should have seen at least 9 full pages
  expect(numFullPages).toBeGreaterThan(8);
});

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
  expect(countsInfo).toEqual([
    {
      function_name: "ask_question",
      max_timestamp: "2025-01-03T21:52:59Z",
      count: 767,
    },
    {
      function_name: "answer_question",
      max_timestamp: "2025-01-03T21:52:59Z",
      count: 764,
    },
    {
      function_name: "generate_secret",
      max_timestamp: "2025-01-03T21:51:29Z",
      count: 50,
    },
    {
      function_name: "dashboard_fixture_write_haiku",
      max_timestamp: "2024-12-25T02:20:45Z",
      count: 494,
    },
    {
      function_name: "dashboard_fixture_extract_entities",
      max_timestamp: "2024-12-18T16:22:43Z",
      count: 400,
    },
  ]);
});
