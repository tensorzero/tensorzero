import { describe, expect, test } from "vitest";
import { tensorZeroClient } from "~/utils/tensorzero.server";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { type JsonInferenceDatapoint } from "~/utils/tensorzero";

describe("update datapoints and make sure the source_inference_id is removed if the input changed", () => {
  test("should remove the source_inference_id if the input changed", async () => {
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
      source_inference_id: null,
    };

    await tensorZeroClient.updateDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
      datapoint,
      true,
    );

    const retrievedDatapoint = await getDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
    );
    expect(retrievedDatapoint?.source_inference_id).toBeNull();
  });

  test("should not remove the source_inference_id if the input did not change", async () => {
    const source_inference_id = "01960843-19be-7dce-922f-b5f618176ec0";
    const datapoint: JsonInferenceDatapoint = {
      function_name: "extract_entities",
      input: {
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO.",
              },
            ],
          },
        ],
      },
      output: {
        person: ["Tim Cook"],
        organization: ["Apple Inc."],
        location: ["Cupertino", "California"],
        miscellaneous: [],
      },

      output_schema: {
        type: "object",
        properties: {
          person: {
            type: "array",
            items: { type: "string" },
          },
          organization: {
            type: "array",
            items: { type: "string" },
          },
          location: {
            type: "array",
            items: { type: "string" },
          },
          miscellaneous: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: ["person", "organization", "location", "miscellaneous"],
        additionalProperties: false,
      },
      tags: {},
      auxiliary: "",
      source_inference_id,
    };

    await tensorZeroClient.updateDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
      datapoint,
      false,
    );

    const retrievedDatapoint = await getDatapoint(
      "test",
      "01960832-7028-743c-8c44-a598aa5130fd",
    );
    expect(retrievedDatapoint?.source_inference_id).toBe(source_inference_id);
  });
});
