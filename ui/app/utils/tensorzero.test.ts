import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { type JsonInferenceDatapoint } from "~/utils/tensorzero";

let tensorZeroClient: ReturnType<typeof getTensorZeroClient>;

describe("update datapoints", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should preserve original source_inference_id and is_custom when updating datapoint", async () => {
    const datapoint: JsonInferenceDatapoint = {
      function_name: "extract_entities",
      name: null,
      episode_id: null,
      staled_at: null,
      input: {
        messages: [
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: "nds] ) :",
              },
            ],
          },
        ],
      },
      output: {
        person: [],
        organization: [],
        location: [],
        miscellaneous: [],
      },
      tags: {},
      auxiliary: "",
      output_schema: {
        $schema: "http://json-schema.org/draft-07/schema#",
        type: "object",
        properties: {
          person: {
            type: "array",
            items: {
              type: "string",
            },
          },
          organization: {
            type: "array",
            items: {
              type: "string",
            },
          },
          location: {
            type: "array",
            items: {
              type: "string",
            },
          },
          miscellaneous: {
            type: "array",
            items: {
              type: "string",
            },
          },
        },
        required: ["person", "organization", "location", "miscellaneous"],
        additionalProperties: false,
      },
      source_inference_id: "01982323-3460-71dd-8cc8-bc4d44a0c88f",
      is_custom: false,
      id: "01960832-7028-743c-8c44-a598aa5130fd",
    };

    await tensorZeroClient.updateDatapoint("test", datapoint);

    const retrievedDatapoint = await tensorZeroClient.getDatapoint(
      "01960832-7028-743c-8c44-a598aa5130fd",
    );
    expect(retrievedDatapoint?.source_inference_id).toBe(
      datapoint.source_inference_id,
    );
    expect(retrievedDatapoint?.is_custom).toBe(false);
  });

  test("should list datapoints", async () => {
    const datapoints = await tensorZeroClient.listDatapoints(
      "foo",
      "extract_entities",
      10,
    );
    expect(datapoints.length).toBe(10);
    for (const datapoint of datapoints) {
      expect(datapoint.function_name).toBe("extract_entities");
    }
  });
});
