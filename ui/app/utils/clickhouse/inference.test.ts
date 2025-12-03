import { expect, test } from "vitest";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import {
  countInferencesForEpisode,
  queryInferenceById,
  listInferencesWithPagination,
  countInferencesByFunction,
  countInferencesForVariant,
  queryModelInferencesByInferenceId,
} from "./inference.server";
import { countInferencesForFunction } from "./inference.server";
import type { ZodTextContent } from "./common";
import { displayModelInferenceInputMessageContentSchema } from "./common";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "~/types/tensorzero";

// Test countInferencesForFunction
test("countInferencesForFunction returns correct counts", async () => {
  const jsonCount = await countInferencesForFunction("extract_entities", {
    type: "json",
    variants: {},
    schemas: {},
    description: "",
    output_schema: { value: {} },
    json_mode_tool_call_config: {
      static_tools_available: [],
      dynamic_tools_available: [],
      provider_tools: [],
      openai_custom_tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
      allowed_tools: { tools: [], choice: "function_default" },
    },
    experimentation: { type: "uniform" },
  });
  expect(jsonCount).toBe(604);

  const chatCount = await countInferencesForFunction("write_haiku", {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
    schemas: {},
    description: "",
    experimentation: { type: "uniform" },
  });
  expect(chatCount).toBe(804);
});

// Test countInferencesForVariant
test("countInferencesForVariant returns correct counts", async () => {
  const jsonCount = await countInferencesForVariant(
    "extract_entities",
    {
      type: "json",
      variants: {},
      schemas: {},
      description: "",
      output_schema: { value: {} },
      json_mode_tool_call_config: {
        static_tools_available: [],
        dynamic_tools_available: [],
        provider_tools: [],
        openai_custom_tools: [],
        tool_choice: "none",
        parallel_tool_calls: false,
        allowed_tools: {
          tools: [],
          choice: "function_default",
        },
      },
      experimentation: { type: "uniform" },
    },
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
      schemas: {},
      description: "",
      experimentation: { type: "uniform" },
    },
    "initial_prompt_gpt4o_mini",
  );
  expect(chatCount).toBe(649);
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
  const output = inference?.output as ContentBlockChatOutput[];
  const firstOutput = output[0] as ZodTextContent;
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
        count: 3,
        function_name: DEFAULT_FUNCTION,
        max_timestamp: "2025-09-08T01:42:25Z",
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

test("displayModelInferenceInputMessageContentSchema accepts thought content blocks", () => {
  // Test thought content block with all fields including summary
  const thoughtContentWithSummary = {
    type: "thought",
    text: "This is a thinking step",
    signature: "abcdef",
    summary: [
      { text: "Summary of the thought", type: "summary_text" },
      { text: "Another summary point", type: "summary_text" },
    ],
    _internal_provider_type: "anthropic",
  };

  const result1 = displayModelInferenceInputMessageContentSchema.safeParse(
    thoughtContentWithSummary,
  );
  expect(result1.success).toBe(true);
  if (result1.success && result1.data.type === "thought") {
    expect(result1.data.text).toBe("This is a thinking step");
    expect(result1.data.summary).toHaveLength(2);
    expect(result1.data.summary?.[0].text).toBe("Summary of the thought");
  }

  // Test thought content block with signature but without summary
  const thoughtContentWithSignature = {
    type: "thought",
    text: "Another thinking step",
    signature: "abcdef",
    _internal_provider_type: "anthropic",
  };

  const result2 = displayModelInferenceInputMessageContentSchema.safeParse(
    thoughtContentWithSignature,
  );
  expect(result2.success).toBe(true);
  if (result2.success && result2.data.type === "thought") {
    expect(result2.data.text).toBe("Another thinking step");
    expect(result2.data.summary).toBeUndefined();
  }

  // Test thought content block with minimal fields (text is null)
  const thoughtContentMinimal = {
    type: "thought",
    text: null,
    signature: "abcdef",
  };

  const result3 = displayModelInferenceInputMessageContentSchema.safeParse(
    thoughtContentMinimal,
  );
  expect(result3.success).toBe(true);
  if (result3.success && result3.data.type === "thought") {
    expect(result3.data.text).toBeUndefined();
    expect(result3.data.summary).toBeUndefined();
  }
});
