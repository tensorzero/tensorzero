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
    tags: {
      scenario: "greeting",
    },
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
    expect(JSON.parse(formData.get("tags") as string)).toEqual(datapoint.tags);
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

    const parsed = parseDatapointFormData(formData);

    expect(parsed).toMatchObject({
      dataset_name: "json-dataset",
      function_name: "extract_entities",
      id: "00000000-0000-0000-0000-000000000010",
      episode_id: "00000000-0000-0000-0000-000000000011",
      input: { messages: [] },
      tags: {},
    });
    expect("output" in parsed).toBe(false);
  });
});
