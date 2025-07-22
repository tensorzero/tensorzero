import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { type JsonInferenceDatapoint } from "~/utils/tensorzero";

let tensorZeroClient: ReturnType<typeof getTensorZeroClient>;

describe("update datapoints", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should preserve original source_inference_id and set is_custom when updating datapoint", async () => {
    const datapoint: JsonInferenceDatapoint = {
      function_name: "extract_entities",
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
      is_custom: true,
    };

    await tensorZeroClient.updateDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
      datapoint,
    );

    const retrievedDatapoint = await getDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
    );
    expect(retrievedDatapoint?.source_inference_id).toBe(
      datapoint.source_inference_id,
    );
    expect(retrievedDatapoint?.is_custom).toBe(true);
  });
});
