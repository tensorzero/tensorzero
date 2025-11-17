import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

let tensorZeroClient: ReturnType<typeof getTensorZeroClient>;

describe("update datapoints", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should create new datapoint with new ID when updating", async () => {
    const result = await tensorZeroClient.updateDatapoint("test", {
      type: "json",
      id: "01960832-7028-743c-8c44-a598aa5130fd",
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
        raw: JSON.stringify({
          person: [],
          organization: [],
          location: [],
          miscellaneous: [],
        }),
      },
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
    });

    // The v1 endpoint creates a new datapoint with a new ID
    expect(result.id).not.toBe("01960832-7028-743c-8c44-a598aa5130fd");

    // Verify the new datapoint exists
    const retrievedDatapoint = await tensorZeroClient.getDatapoint(result.id);
    expect(retrievedDatapoint).not.toBeNull();
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
