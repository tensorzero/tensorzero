import {
  type ParsedJsonInferenceDatapointRow,
  type ParsedChatInferenceDatapointRow,
} from "./datasets";
import {
  countDatapointsForDatasetFunction,
  getAdjacentDatapointIds,
  getDatapoint,
  getNumberOfDatasets,
  insertDatapoint,
  insertRowsForDataset,
  staleDatapoint,
} from "./datasets.server";
import { expect, test, describe } from "vitest";
import { v7 as uuid } from "uuid";
import { getClickhouseClient } from "./client.server";

describe("getNumberOfDatasets", () => {
  test("returns the correct number of datasets", async () => {
    const count = await getNumberOfDatasets();
    // This should be equal to 3 in the fixtures but since we want to be able to re-run this test
    // and run it in parallel to the other tests which add datasets to the DB,
    // we use a greater than or equal check.
    expect(count).toBeGreaterThanOrEqual(3);
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
      name: null,
      id: "01942e26-c48c-7720-b971-a1f7a3a9ac98",
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                text: "Is it a living thing?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                text: "no.",
              },
            ],
            role: "assistant",
          },
          {
            content: [
              {
                type: "text",
                text: "Is it commonly found indoors?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                text: "no.",
              },
            ],
            role: "assistant",
          },
          {
            content: [
              {
                type: "text",
                text: "Is it a natural object, like a rock or tree?",
              },
            ],
            role: "user",
          },
          {
            content: [
              {
                type: "text",
                text: "yes.",
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
      staled_at: null,
      updated_at: "2025-02-19T00:26:06Z",
      source_inference_id: null,
      is_custom: false,
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
      name: null,
      id: "01934fc5-ea98-71f0-8191-9fd88f34c28b",
      input: {
        messages: [
          {
            content: [
              {
                type: "template",
                name: "user",
                arguments: {
                  topic: "upward",
                },
              },
            ],
            role: "user",
          },
        ],
      },
      is_deleted: false,
      staled_at: null,
      output: [
        {
          text: 'Alright, the theme of "upward" immediately brings to mind things that ascend or rise. This can be movements, emotions, or natural events.\n\nLet\'s craft a haiku:\n\nMountains touch the sky,  \nClouds race past the soaring peaks,  \nWorld beneath grows small.',
          type: "text",
        },
      ],
      tags: {},
      updated_at: "2025-02-19T00:25:04Z",
      source_inference_id: null,
      tool_params: undefined,
      is_custom: false,
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
    const datapoint_id = uuid();
    const source_inference_id = uuid();
    const chatDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: "test_chat_dataset",
      function_name: "write_haiku",
      id: datapoint_id,
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff9",
      name: null,
      input: {
        messages: [
          {
            content: [
              {
                type: "template",
                name: "user",
                arguments: {
                  topic: "testing",
                },
              },
            ],
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
      staled_at: null,
      source_inference_id,
      is_custom: false,
    };

    // Test insertion
    await insertDatapoint(chatDatapoint);

    // Test retrieval
    const retrievedDatapoint = await getDatapoint(
      "test_chat_dataset",
      datapoint_id,
    );
    expect(retrievedDatapoint).toBeTruthy();
    expect(retrievedDatapoint?.id).toBe(chatDatapoint.id);
    expect(retrievedDatapoint?.function_name).toBe(chatDatapoint.function_name);
    expect(retrievedDatapoint?.dataset_name).toBe(chatDatapoint.dataset_name);
    expect(retrievedDatapoint?.input).toEqual(chatDatapoint.input);
    expect(retrievedDatapoint?.source_inference_id).toBe(source_inference_id);
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

    // Test staling
    await staleDatapoint(chatDatapoint.dataset_name, chatDatapoint.id, "chat");
    // Sleep 100ms
    await new Promise((resolve) => setTimeout(resolve, 1000));
    // Try and get the datapoint
    const staled_getter_result = await getDatapoint(
      "test_chat_dataset",
      datapoint_id,
    );
    // Test that the datapoint was properly staled
    expect(staled_getter_result).toBeNull();

    // Also check that if we force it to allow stale then it is retrieved
    const staledDatapoint = await getDatapoint(
      "test_chat_dataset",
      datapoint_id,
      true,
    );
    expect(staledDatapoint).toBeDefined();
    expect(staledDatapoint?.id).toEqual(datapoint_id);
    expect(staledDatapoint?.staled_at).toBeDefined();
  });

  test("json datapoint lifecycle - insert, get, delete", async () => {
    const datapoint_id = uuid();
    const source_inference_id = uuid();
    const jsonDatapoint: ParsedJsonInferenceDatapointRow = {
      dataset_name: "test_json_dataset",
      function_name: "extract_entities",
      id: datapoint_id,
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff8",
      name: null,
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                text: "Extract entities from: John visited Paris",
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
      staled_at: null,
      source_inference_id,
      is_custom: false,
    };

    // Test insertion
    await insertDatapoint(jsonDatapoint);

    // Test retrieval
    const retrievedDatapoint = await getDatapoint(
      "test_json_dataset",
      datapoint_id,
    );
    expect(retrievedDatapoint).toBeTruthy();
    expect(retrievedDatapoint?.id).toBe(jsonDatapoint.id);
    expect(retrievedDatapoint?.function_name).toBe(jsonDatapoint.function_name);
    expect(retrievedDatapoint?.dataset_name).toBe(jsonDatapoint.dataset_name);
    expect(retrievedDatapoint?.input).toEqual(jsonDatapoint.input);
    expect(retrievedDatapoint?.output).toEqual(jsonDatapoint.output);

    // Check if it's a JSON inference row before accessing output_schema
    if (retrievedDatapoint && "output_schema" in retrievedDatapoint) {
      expect(JSON.stringify(retrievedDatapoint.output_schema)).toBe(
        JSON.stringify(jsonDatapoint.output_schema),
      );
    } else {
      throw new Error("Expected JSON inference row but got chat inference row");
    }

    // Test deletion
    await staleDatapoint("test_json_dataset", datapoint_id, "json");
    // Sleep 100ms
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const deletedDatapoint = await getDatapoint(
      "test_json_dataset",
      datapoint_id,
    );
    expect(deletedDatapoint).toBeNull();

    // Also check that if we force it to allow stale then it is retrieved
    const staledDatapoint = await getDatapoint(
      "test_json_dataset",
      datapoint_id,
      true,
    );
    expect(staledDatapoint).toBeDefined();
    expect(staledDatapoint?.id).toEqual(datapoint_id);
    expect(staledDatapoint?.staled_at).toBeDefined();
  });

  test("handles non-existent datapoint retrieval", async () => {
    const nonExistentDatapoint = await getDatapoint(
      "non_existent_dataset",
      "01934fc5-ea98-71f0-8191-9fd88f34c30d",
    );
    expect(nonExistentDatapoint).toBeNull();
  });

  test("handles duplicate insertions gracefully", async () => {
    const source_inference_id = uuid();
    const chatDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: "test_chat_dataset",
      function_name: "write_haiku",
      id: "01934fc5-ea98-71f0-8191-9fd88f34c31e",
      episode_id: "0193fb9d-73ad-7ad2-807d-a2ef10088ff7",
      name: null,
      input: {
        messages: [
          {
            content: [
              {
                type: "text",
                text: "Write a haiku about duplicates",
              },
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
      staled_at: null,
      source_inference_id,
      is_custom: false,
    };

    // First insertion
    await insertDatapoint(chatDatapoint);

    // Second insertion with same ID should not throw
    await insertDatapoint(chatDatapoint);

    // Cleanup
    await staleDatapoint(chatDatapoint.dataset_name, chatDatapoint.id, "chat");
    // Sleep 100ms
    // Sleep 100ms
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const deletedDatapoint = await getDatapoint(
      "test_json_dataset",
      chatDatapoint.id,
    );
    expect(deletedDatapoint).toBeNull();
  });

  test("handles staling of non-existent datapoint", async () => {
    // Should not throw
    await expect(staleDatapoint("fake", uuid(), "chat")).resolves.not.toThrow();
  });
});

describe("countDatapointsForDatasetFunction", () => {
  test("returns the correct count for a dataset and chat function", async () => {
    const count = await countDatapointsForDatasetFunction("foo", "write_haiku");
    expect(count).toBe(78);
  });
  test("returns the correct count for a dataset and json function", async () => {
    const count = await countDatapointsForDatasetFunction(
      "foo",
      "extract_entities",
    );
    expect(count).toBe(43);
  });
  test("returns 0 for a non-existent dataset and real function", async () => {
    const count = await countDatapointsForDatasetFunction(
      "fake",
      "write_haiku",
    );
    expect(count).toBe(0);
  });
  test("returns 0 for a real dataset and non-existent function", async () => {
    const count = await countDatapointsForDatasetFunction("foo", "fake");
    expect(count).toBeNull();
  });
});

describe("insertRowsForDataset", () => {
  test("handles invalid dataset names", async () => {
    await expect(
      insertRowsForDataset({
        dataset_name: "builder",
        inference_type: "chat",
        output_source: "none",
      }),
    ).rejects.toThrow();
  });

  test("correctly handles incremental insertions for json", async () => {
    // Generate a random dataset name so this test can be safely re-run
    const dataset_name = `test_incremental_insertions_${uuid()}`;
    const insert_with_cutoff = async (cutoff: number) => {
      const rowsAdded = await insertRowsForDataset({
        dataset_name,
        inference_type: "json",
        function_name: "extract_entities",
        metric_filter: {
          metric: "jaccard_similarity",
          metric_type: "float",
          operator: ">",
          threshold: cutoff,
          join_on: "inference",
        },
        output_source: "none",
      });
      return rowsAdded;
    };
    const rowsAdded = await insert_with_cutoff(0.9);
    expect(rowsAdded).toBe(54);
    const rowsAdded2 = await insert_with_cutoff(0.8);
    expect(rowsAdded2).toBe(0);
    const rowsAdded3 = await insert_with_cutoff(0.7);
    expect(rowsAdded3).toBe(5);
    // Try a different output source (this should not write any rows)
    const rowsAdded4 = await insertRowsForDataset({
      dataset_name,
      inference_type: "json",
      function_name: "extract_entities",
      metric_filter: {
        metric: "jaccard_similarity",
        metric_type: "float",
        operator: ">",
        threshold: 0.7,
        join_on: "inference",
      },
      output_source: "inference",
    });
    expect(rowsAdded4).toBe(0);
  }, 10000); // 10 second timeout

  test("correctly handles incremental insertions for chat", async () => {
    // Generate a random dataset name so this test can be safely re-run
    const dataset_name = `test_incremental_insertions_${uuid()}`;
    const insert_with_cutoff = async (cutoff: number) => {
      const rowsAdded = await insertRowsForDataset({
        dataset_name,
        inference_type: "chat",
        function_name: "write_haiku",
        metric_filter: {
          metric: "haiku_rating",
          metric_type: "float",
          operator: ">",
          threshold: cutoff,
          join_on: "inference",
        },
        output_source: "none",
      });
      return rowsAdded;
    };
    const rowsAdded = await insert_with_cutoff(0.9);
    expect(rowsAdded).toBe(57);
    const rowsAdded2 = await insert_with_cutoff(0.8);
    expect(rowsAdded2).toBe(10);
    const rowsAdded3 = await insert_with_cutoff(0.7);
    expect(rowsAdded3).toBe(8);

    // Try a different output source (this should not write any rows)
    const rowsAdded4 = await insertRowsForDataset({
      dataset_name,
      inference_type: "chat",
      function_name: "write_haiku",
      metric_filter: {
        metric: "haiku_rating",
        metric_type: "float",
        operator: ">",
        threshold: 0.7,
        join_on: "inference",
      },
      output_source: "inference",
    });
    expect(rowsAdded4).toBe(0);
  }, 10000); // 10 second timeout
});

describe("insertDatapoint", () => {
  test("handles invalid dataset names", async () => {
    await expect(
      insertDatapoint({
        dataset_name: "builder",
        function_name: "write_haiku",
        id: uuid(),
        episode_id: null,
        name: null,
        input: { messages: [] },
        output: [],
        tool_params: {},
        tags: {},
        auxiliary: "",
        updated_at: new Date().toISOString(),
        is_deleted: false,
        staled_at: null,
        source_inference_id: null,
        is_custom: true,
      }),
    ).rejects.toThrow();
  });
});

describe("getAdjacentDatapointIds", () => {
  test("returns the correct adjacent datapoint ids", async () => {
    const adjacentIds = await getAdjacentDatapointIds(
      "foo",
      "01934fc5-ea98-71f0-8191-9fd88f34c28b",
    );
    expect(adjacentIds).toEqual({
      next_id: "0193514c-ec40-7911-ad63-460bb9c861e1",
      previous_id: "01934fbc-3250-7571-ad38-041f27ffa3f9",
    });
  });

  test("returns null for next_id if there is no next datapoint", async () => {
    const resultSet = await getClickhouseClient().query({
      query: `SELECT uint_to_uuid(max(id_uint)) as id FROM
      ( SELECT toUInt128(id) as id_uint FROM ChatInferenceDatapoint WHERE dataset_name={dataset_name:String} AND staled_at IS NULL
       UNION ALL
       SELECT toUInt128(id) as id_uint FROM JsonInferenceDatapoint WHERE dataset_name={dataset_name:String} AND staled_at IS NULL
      ) LIMIT 1`,
      format: "JSON",
      query_params: { dataset_name: "foo" },
    });
    const lastFooDatapointId = await resultSet.json<{ id: string }>();
    const adjacentIds = await getAdjacentDatapointIds(
      "foo",
      lastFooDatapointId.data[0].id,
    );
    expect(adjacentIds).toEqual({
      previous_id: "0196374a-d03f-7420-9da5-1561cba71ddb",
      next_id: null,
    });
  });

  test("returns null for previous_id if there is no previous datapoint", async () => {
    const resultSet = await getClickhouseClient().query({
      query: `SELECT uint_to_uuid(min(id_uint)) as id FROM
      ( SELECT toUInt128(id) as id_uint FROM ChatInferenceDatapoint WHERE dataset_name={dataset_name:String} AND staled_at IS NULL
       UNION ALL
       SELECT toUInt128(id) as id_uint FROM JsonInferenceDatapoint WHERE dataset_name={dataset_name:String} AND staled_at IS NULL
      ) LIMIT 1`,
      format: "JSON",
      query_params: { dataset_name: "foo" },
    });
    const firstFooDatapointId = await resultSet.json<{ id: string }>();
    const adjacentIds = await getAdjacentDatapointIds(
      "foo",
      firstFooDatapointId.data[0].id,
    );
    expect(adjacentIds).toEqual({
      previous_id: null,
      next_id: "01934ef3-d558-79d3-896e-c5d9fb89103d",
    });
  });
});
