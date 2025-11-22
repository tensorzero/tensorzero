import { describe, expect, test } from "vitest";
import {
  serializeUpdateDatapointToFormData,
  parseUpdateDatapointFormData,
  type UpdateDatapointFormData,
} from "./formDataUtils";

function createChatDatapoint(): Omit<UpdateDatapointFormData, "action"> {
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

describe("serializeUpdateDatapointToFormData", () => {
  test("serializes complex datapoint fields while omitting null entries", () => {
    const datapoint = createChatDatapoint();

    const formData = serializeUpdateDatapointToFormData(datapoint);

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
    expect(formData.get("action")).toBe("update");
  });
});

describe("parseUpdateDatapointFormData", () => {
  test("round-trips a serialized chat datapoint", () => {
    const datapoint = createChatDatapoint();

    const parsed = parseUpdateDatapointFormData(
      serializeUpdateDatapointToFormData(datapoint),
    );

    expect(parsed).toEqual({ ...datapoint, action: "update" });
  });

  test("parses a JSON inference datapoint with optional fields omitted", () => {
    const formData = new FormData();
    formData.set("dataset_name", "json-dataset");
    formData.set("function_name", "extract_entities");
    formData.set("id", "00000000-0000-0000-0000-000000000010");
    formData.set("episode_id", "00000000-0000-0000-0000-000000000011");
    formData.set("input", JSON.stringify({ messages: [] }));
    formData.set("tags", JSON.stringify({}));
    formData.set("action", "update");

    const parsed = parseUpdateDatapointFormData(formData);

    expect(parsed).toMatchObject({
      dataset_name: "json-dataset",
      function_name: "extract_entities",
      id: "00000000-0000-0000-0000-000000000010",
      episode_id: "00000000-0000-0000-0000-000000000011",
      input: { messages: [] },
      tags: {},
      action: "update",
    });
    expect("output" in parsed).toBe(false);
  });
});
