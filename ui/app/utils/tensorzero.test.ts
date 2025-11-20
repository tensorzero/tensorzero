import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

let tensorZeroClient: ReturnType<typeof getTensorZeroClient>;

describe("update datapoints", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should preserve source_inference_id, generate a new ID, and set is_custom to true when updating a datapoint", async () => {
    // Create datapoint from inference
    const inferenceId = "0196368e-5505-7721-88d2-644a2da892a7";
    const createResult =
      await tensorZeroClient.createDatapointFromInferenceLegacy(
        "test",
        inferenceId,
        "inherit",
        "extract_entities",
        "gpt4o_initial_prompt",
        "0196368e-5505-7721-88d2-645619b42142",
      );

    // Verify initial state: is_custom should be false, source_inference_id should match
    const initialDatapoint = await tensorZeroClient.getDatapoint(
      createResult.id,
    );
    expect(initialDatapoint).not.toBeNull();
    expect(initialDatapoint?.is_custom).toBe(false);
    expect(initialDatapoint?.source_inference_id).toBe(inferenceId);

    // TypeScript refinement: we've verified initialDatapoint is not null
    if (!initialDatapoint || initialDatapoint.type !== "json") {
      throw new Error("Expected JSON datapoint");
    }

    // Update the datapoint (e.g., modify the output)
    const updatedOutput = {
      person: ["Updated Person"],
      organization: ["Updated Org"],
      location: ["Updated Location"],
      miscellaneous: ["Updated Misc"],
    };

    const updateResult = await tensorZeroClient.updateDatapoint("test", {
      type: "json",
      id: initialDatapoint.id,
      // TODO (#4674 #4675): fix this casting
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      input: initialDatapoint.input as any, // Type conversion needed: StoredInput has ObjectStoragePointer files, Input expects full file types
      output: {
        raw: JSON.stringify(updatedOutput),
      },
      output_schema: initialDatapoint.output_schema,
    });

    // Verify updated state
    const updatedDatapoint = await tensorZeroClient.getDatapoint(
      updateResult.id,
    );

    // New ID should be created
    expect(updateResult.id).not.toBe(createResult.id);

    // source_inference_id should be preserved
    expect(updatedDatapoint?.source_inference_id).toBe(inferenceId);

    // is_custom should now be true (custom modification)
    expect(updatedDatapoint?.is_custom).toBe(true);
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
