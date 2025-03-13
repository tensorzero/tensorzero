import {
  DatasetQueryParamsSchema,
  type DatasetDetailRow,
  type ParsedJsonInferenceDatapointRow,
  type ParsedChatInferenceDatapointRow,
} from "./datasets";
import {
  countRowsForDataset,
  getDatapoint,
  getDatasetCounts,
  getDatasetRows,
  insertDatapoint,
  deleteDatapoint,
} from "./datasets.server";
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

describe("getDatasetCounts", () => {
  test("returns the correct counts for all datasets", async () => {
    const counts = await getDatasetCounts();
    expect(counts).toEqual(
      // We only assert that the result contains the expected datasets
      // Because other tests insert into the table, there could be additional datasets
      expect.arrayContaining([
        {
          count: 5,
          dataset_name: "bar",
          last_updated: "2025-02-19T00:26:06Z",
        },
        {
          count: 116,
          dataset_name: "foo",
          last_updated: "2025-02-19T00:25:29Z",
        },
      ]),
    );
  });
});

describe("getDatasetRows", () => {
  test("returns the correct rows for a specific dataset", async () => {
    const rows = await getDatasetRows("notadataset", 10, 0);
    expect(rows).toEqual([]);
  });
  test("paging through the rows of foo", async () => {
    let allRows: DatasetDetailRow[] = [];
    let offset = 0;
    const pageSize = 10;

    while (true) {
      const rows = await getDatasetRows("foo", pageSize, offset);
      allRows = [...allRows, ...rows];
      offset += pageSize;
      if (rows.length !== pageSize) break;
    }

    expect(allRows.length).toBe(116);
    expect(allRows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: expect.any(String),
          type: expect.stringMatching(/^(chat|json)$/),
          function_name: expect.any(String),
          episode_id: expect.any(String),
          updated_at: expect.any(String),
        }),
      ]),
    );
  });
  test("paging through bar dataset", async () => {
    let allRows: DatasetDetailRow[] = [];
    let offset = 0;
    const pageSize = 10;

    while (true) {
      const rows = await getDatasetRows("bar", pageSize, offset);
      allRows = [...allRows, ...rows];
      offset += pageSize;
      if (rows.length !== pageSize) break;
    }

    expect(allRows.length).toBe(5);
    expect(allRows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: expect.any(String),
          type: "json",
          function_name: expect.any(String),
          episode_id: expect.any(String),
          updated_at: expect.any(String),
        }),
      ]),
    );
    // Verify all rows are json type
    expect(allRows.every((row) => row.type === "json")).toBe(true);
  });
});

describe("getDatapoint", () => {
  test("returns the correct datapoint for a specific dataset (json)", async () => {
    const datapoint = await getDatapoint(
      "bar",
      "01942e26-c48c-7720-b971-a1f7a3a9ac98",
    );
    expect(datapoint).toEqual({
      auxiliary: "",
      dataset_name: "bar",
      episode_id: "01942e26-4693-7e80-8591-47b98e25d721",
      function_name: "ask_question",
      id: "01942e26-c48c-7720-b971-a1f7a3a9ac98",
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                value: "Is it a living thing?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                value: "no.",
              },
            ],
            role: "assistant",
          },
          {
            content: [
              {
                type: "text",
                value: "Is it commonly found indoors?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                value: "no.",
              },
            ],
            role: "assistant",
          },
          {
            content: [
              {
                type: "text",
                value: "Is it a natural object, like a rock or tree?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                value: "yes.",
              },
            ],
            role: "assistant",
          },
        ],
        system: {
          remaining_questions: 18,
        },
      },
      is_deleted: false,
      output: {
        parsed: {
          question: "Is it a large natural object, like a mountain or a tree?",
          thinking:
            "Since the object is not a living thing and is not commonly found indoors, but is a natural object, it narrows down the possibilities to various elements from nature. It could be a rock, a tree, or potentially something like a mountain or a river. To further narrow it down, I will ask if it is a large object or a small object.",
        },
        raw: `{
  "thinking": "Since the object is not a living thing and is not commonly found indoors, but is a natural object, it narrows down the possibilities to various elements from nature. It could be a rock, a tree, or potentially something like a mountain or a river. To further narrow it down, I will ask if it is a large object or a small object.",
  "question": "Is it a large natural object, like a mountain or a tree?"
}`,
      },
      output_schema: {
        additionalProperties: false,
        properties: {
          question: {
            type: "string",
          },
          thinking: {
            type: "string",
          },
        },
        required: ["thinking", "question"],
        type: "object",
      },
      tags: {},
      updated_at: "2025-02-19T00:26:06Z",
    });
  });

  test("returns the correct datapoint for a specific dataset (chat)", async () => {
    const datapoint = await getDatapoint(
      "foo",
      "01934fc5-ea98-71f0-8191-9fd88f34c28b",
    );
    expect(datapoint).toEqual({
      auxiliary: "",
      dataset_name: "foo",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff9",
      function_name: "write_haiku",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c28b",
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                value: {
                  topic: "upward",
                },
              },
            ],
            role: "user",
          },
        ],
      },
      is_deleted: false,
      output: [
        {
          text: 'Alright, the theme of "upward" immediately brings to mind things that ascend or rise. This can be movements, emotions, or natural events.\n\nLet\'s craft a haiku:\n\nMountains touch the sky,  \nClouds race past the soaring peaks,  \nWorld beneath grows small.',
          type: "text",
        },
      ],
      tags: {},
      tool_params: {},
      updated_at: "2025-02-19T00:25:04Z",
    });
  });

  test("empty result", async () => {
    const datapoint = await getDatapoint(
      "foo",
      "00000000-0000-0000-0000-000000000000",
    );
    expect(datapoint).toEqual(null);
  });
});

describe("datapoint operations", () => {
  test("chat datapoint lifecycle - insert, get, delete", async () => {
    const chatDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: "test_chat_dataset",
      function_name: "write_haiku",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c28b",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff9",
      input: {
        messages: [
          {
            content: [{ type: "text", value: "Write a haiku about testing" }],
            role: "user" as const,
          },
        ],
      },
      output: [
        {
          type: "text",
          text: "Code flows like water\nTests catch bugs in their net now\nPeace in the program",
        },
      ],
      tool_params: {},
      tags: {},
      auxiliary: "",
      updated_at: new Date().toISOString(),
      is_deleted: false,
    };

    // Test insertion
    await insertDatapoint(chatDatapoint);

    // Test retrieval
    const retrievedDatapoint = await getDatapoint(
      "test_chat_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c28b",
    );
    expect(retrievedDatapoint).toBeTruthy();
    expect(retrievedDatapoint?.id).toBe(chatDatapoint.id);
    expect(retrievedDatapoint?.function_name).toBe(chatDatapoint.function_name);
    expect(retrievedDatapoint?.dataset_name).toBe(chatDatapoint.dataset_name);
    expect(retrievedDatapoint?.input).toEqual(chatDatapoint.input);

    // Check if it's a chat inference row before accessing tool_params
    if (retrievedDatapoint && "tool_params" in retrievedDatapoint) {
      expect(JSON.stringify(retrievedDatapoint.output)).toBe(
        JSON.stringify(chatDatapoint.output),
      );
      expect(JSON.stringify(retrievedDatapoint.tool_params)).toBe(
        JSON.stringify(chatDatapoint.tool_params),
      );
    } else {
      throw new Error("Expected chat inference row but got JSON inference row");
    }

    // Test deletion
    await deleteDatapoint(chatDatapoint);
    const deletedDatapoint = await getDatapoint(
      "test_chat_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c28b",
    );
    expect(deletedDatapoint).toBeNull();
  });

  test("json datapoint lifecycle - insert, get, delete", async () => {
    const jsonDatapoint: ParsedJsonInferenceDatapointRow = {
      dataset_name: "test_json_dataset",
      function_name: "extract_entities",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c29c",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff8",
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                value: "Extract entities from: John visited Paris",
              },
            ],
            role: "user" as const,
          },
        ],
      },
      output: {
        raw: JSON.stringify({
          entities: ["John", "Paris"],
          types: ["PERSON", "LOCATION"],
        }),
        parsed: {
          entities: ["John", "Paris"],
          types: ["PERSON", "LOCATION"],
        },
      },
      output_schema: {
        type: "object",
        properties: {
          entities: { type: "array", items: { type: "string" } },
          types: { type: "array", items: { type: "string" } },
        },
        required: ["entities", "types"],
      },
      tags: {},
      auxiliary: "",
      updated_at: new Date().toISOString(),
      is_deleted: false,
    };

    // Test insertion
    await insertDatapoint(jsonDatapoint);

    // Test retrieval
    const retrievedDatapoint = await getDatapoint(
      "test_json_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c29c",
    );
    expect(retrievedDatapoint).toBeTruthy();
    expect(retrievedDatapoint?.id).toBe(jsonDatapoint.id);
    expect(retrievedDatapoint?.function_name).toBe(jsonDatapoint.function_name);
    expect(retrievedDatapoint?.dataset_name).toBe(jsonDatapoint.dataset_name);
    expect(retrievedDatapoint?.input).toEqual(jsonDatapoint.input);
    expect(retrievedDatapoint?.output).toEqual(jsonDatapoint.output);

    // Check if it's a JSON inference row before accessing output_schema
    if (retrievedDatapoint && !("tool_params" in retrievedDatapoint)) {
      expect(JSON.stringify(retrievedDatapoint.output_schema)).toBe(
        JSON.stringify(jsonDatapoint.output_schema),
      );
    } else {
      throw new Error("Expected JSON inference row but got chat inference row");
    }

    // Test deletion
    await deleteDatapoint(jsonDatapoint);
    const deletedDatapoint = await getDatapoint(
      "test_json_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c29c",
    );
    expect(deletedDatapoint).toBeNull();
  });

  test("handles non-existent datapoint retrieval", async () => {
    const nonExistentDatapoint = await getDatapoint(
      "non_existent_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c30d",
    );
    expect(nonExistentDatapoint).toBeNull();
  });

  test("handles duplicate insertions gracefully", async () => {
    const chatDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: "test_chat_dataset",
      function_name: "write_haiku",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c31e",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff7",
      input: {
        messages: [
          {
            content: [
              { type: "text", value: "Write a haiku about duplicates" },
            ],
            role: "user" as const,
          },
        ],
      },
      output: [
        {
          type: "text",
          text: "Copies everywhere\nDuplicates fill the database\nUnique keys break down",
        },
      ],
      tool_params: {},
      tags: {},
      auxiliary: "",
      updated_at: new Date().toISOString(),
      is_deleted: false,
    };

    // First insertion
    await insertDatapoint(chatDatapoint);

    // Second insertion with same ID should not throw
    await insertDatapoint(chatDatapoint);

    // Cleanup
    await deleteDatapoint(chatDatapoint);
  });

  test("handles deletion of non-existent datapoint", async () => {
    const nonExistentDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: "non_existent_dataset",
      function_name: "non_existent_function",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c32f",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff6",
      input: {
        messages: [
          {
            role: "user" as const,
            content: [{ type: "text", value: "test" }],
          },
        ],
      },
      output: [{ type: "text", text: "test" }],
      tool_params: {},
      tags: {},
      auxiliary: "",
      updated_at: new Date().toISOString(),
      is_deleted: false,
    };

    // Should not throw
    await expect(deleteDatapoint(nonExistentDatapoint)).resolves.not.toThrow();
  });
});
