import { describe, expect, test } from "vitest";
import {
  serializeDatapointToFormData,
  parseDatapointFormData,
  type DatapointFormData,
} from "./formDataUtils";

function createChatDatapoint(): DatapointFormData {
  return {
    dataset_name: "chat-dataset",
    function_name: "reply",
    name: undefined,
    id: "00000000-0000-0000-0000-000000000001",
    episode_id: "00000000-0000-0000-0000-000000000002",
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "How are you?",
            },
          ],
        },
      ],
    },
    output: [
      {
        type: "text",
        text: "I'm doing well, thanks for asking!",
      },
    ],
    // @ts-expect-error tool_params is not in the DatapointFormData type because it's a union. We should fix this when moving to use napi-rs types.
    tool_params: {
      temperature: 0.7,
      top_p: 0.9,
    },
    tags: {
      scenario: "greeting",
    },
    auxiliary: "conversation metadata",
    is_deleted: false,
    updated_at: "2024-01-01T00:00:00.000Z",
    staled_at: null,
    source_inference_id: "00000000-0000-0000-0000-000000000099",
    is_custom: true,
  };
}

describe("serializeDatapointToFormData", () => {
  test("serializes complex datapoint fields while omitting null entries", () => {
    const datapoint = createChatDatapoint();

    const formData = serializeDatapointToFormData(datapoint);

    expect(formData.get("dataset_name")).toBe(datapoint.dataset_name);
    expect(formData.get("function_name")).toBe(datapoint.function_name);
    expect(formData.get("id")).toBe(datapoint.id);
    expect(formData.get("episode_id")).toBe(datapoint.episode_id);
    expect(JSON.parse(formData.get("input") as string)).toEqual(
      datapoint.input,
    );
    expect(JSON.parse(formData.get("output") as string)).toEqual(
      datapoint.output,
    );
    expect(JSON.parse(formData.get("tool_params") as string)).toEqual(
      // @ts-expect-error tool_params is not in the DatapointFormData type because it's a union. We should fix this when moving to use napi-rs types.
      datapoint.tool_params,
    );
    expect(JSON.parse(formData.get("tags") as string)).toEqual(datapoint.tags);
    expect(formData.get("auxiliary")).toBe(datapoint.auxiliary);
    expect(formData.get("is_deleted")).toBe("false");
    expect(formData.get("updated_at")).toBe(datapoint.updated_at);
    expect(formData.get("source_inference_id")).toBe(
      datapoint.source_inference_id,
    );
    expect(formData.get("is_custom")).toBe("true");
    expect(formData.has("name")).toBe(false);
    expect(formData.has("staled_at")).toBe(false);
  });
});

describe("parseDatapointFormData", () => {
  test("round-trips a serialized chat datapoint", () => {
    const datapoint = createChatDatapoint();

    const parsed = parseDatapointFormData(
      serializeDatapointToFormData(datapoint),
    );

    expect(parsed).toEqual(datapoint);
  });

  test("parses a JSON inference datapoint with optional fields omitted", () => {
    const formData = new FormData();
    formData.set("dataset_name", "json-dataset");
    formData.set("function_name", "extract_entities");
    formData.set("id", "00000000-0000-0000-0000-000000000010");
    formData.set("episode_id", "00000000-0000-0000-0000-000000000011");
    formData.set("input", JSON.stringify({ messages: [] }));
    formData.set("tags", JSON.stringify({}));
    formData.set(
      "output_schema",
      JSON.stringify({
        type: "object",
        properties: {
          entities: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: ["entities"],
      }),
    );
    formData.set("auxiliary", "additional context");
    formData.set("is_deleted", "false");
    formData.set("updated_at", "2024-02-01T12:00:00.000Z");
    formData.set("source_inference_id", "00000000-0000-0000-0000-000000000012");
    formData.set("is_custom", "true");

    const parsed = parseDatapointFormData(formData);

    expect(parsed).toMatchObject({
      dataset_name: "json-dataset",
      function_name: "extract_entities",
      id: "00000000-0000-0000-0000-000000000010",
      episode_id: "00000000-0000-0000-0000-000000000011",
      input: { messages: [] },
      tags: {},
      output_schema: {
        type: "object",
        properties: {
          entities: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: ["entities"],
      },
      auxiliary: "additional context",
      is_deleted: false,
      updated_at: "2024-02-01T12:00:00.000Z",
      source_inference_id: "00000000-0000-0000-0000-000000000012",
      is_custom: true,
    });
    expect(parsed.staled_at).toBeNull();
    expect("output" in parsed).toBe(false);
  });
});
