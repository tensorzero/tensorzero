import { describe, expect, test, beforeEach, vi } from "vitest";
import { v7 as uuid } from "uuid";
import {
  deleteDatapoint,
  saveDatapoint,
  renameDatapoint,
} from "./datapointOperations.server";
import type {
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "~/utils/clickhouse/datasets";
import type { ZodDatapoint } from "~/utils/tensorzero";
import type {
  GetDatasetMetadataParams,
  DatasetMetadata,
  UpdateDatapointsMetadataRequest,
} from "~/types/tensorzero";

// TODO(shuyangli): Once we remove all custom logic from the Node client, make mocking more ergonomic by providing a mock client at the tensorzero-node level.

// Mock TensorZero client at the module boundary
const mockUpdateDatapoint = vi.fn(
  async (_datasetName: string, datapoint: ZodDatapoint) => ({
    id: datapoint.id,
  }),
);
const mockUpdateDatapointsMetadata = vi.fn(
  async (_datasetName: string, _request: UpdateDatapointsMetadataRequest) => ({
    ids: [],
  }),
);
const mockDeleteDatapoints = vi.fn(
  async (_datasetName: string, _datapointIds: string[]) => ({
    num_deleted_datapoints: BigInt(_datapointIds.length),
  }),
);
vi.mock("~/utils/tensorzero.server", () => ({
  getTensorZeroClient: vi.fn(() => ({
    updateDatapoint: mockUpdateDatapoint,
    updateDatapointsMetadata: mockUpdateDatapointsMetadata,
    deleteDatapoints: mockDeleteDatapoints,
  })),
}));

// Mock the datasets server functions
const mockGetDatasetMetadata = vi.hoisted(() =>
  vi.fn<(params: GetDatasetMetadataParams) => Promise<DatasetMetadata[]>>(
    async () => [],
  ),
);
vi.mock("~/utils/clickhouse/datasets.server", () => ({
  getDatasetMetadata: mockGetDatasetMetadata,
}));

describe("datapointOperations", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
  });

  describe("deleteDatapoint", () => {
    test("should call deleteDatapoints and redirect to /datasets when dataset is empty", async () => {
      // Mock getDatasetMetadata to return empty array (no datasets)
      vi.mocked(mockGetDatasetMetadata).mockResolvedValueOnce([]);

      const datasetName = "nonexistent_dataset";
      const datapointId = uuid();

      const result = await deleteDatapoint({
        dataset_name: datasetName,
        id: datapointId,
      });

      expect(mockDeleteDatapoints).toHaveBeenCalledWith(datasetName, [
        datapointId,
      ]);
      expect(result.redirectTo).toBe("/datasets");
    });

    test("should redirect to dataset page when dataset still has datapoints", async () => {
      const datasetName = "foo";
      const datapointId = uuid();

      // Mock getDatasetMetadata to return a dataset with count
      vi.mocked(mockGetDatasetMetadata).mockResolvedValueOnce([
        {
          dataset_name: datasetName,
          count: 10,
          last_updated: "2025-04-15T02:33:58Z",
        },
      ]);

      const result = await deleteDatapoint({
        dataset_name: datasetName,
        id: datapointId,
      });

      expect(mockDeleteDatapoints).toHaveBeenCalledWith(datasetName, [
        datapointId,
      ]);
      expect(result.redirectTo).toBe(`/datasets/${datasetName}`);
    });
  });

  describe("saveDatapoint - chat", () => {
    test("should generate new ID, call updateDatapoint, and stale old datapoint", async () => {
      const datasetName = "test_dataset";
      const originalId = uuid();
      const sourceInferenceId = uuid();
      const episodeId = uuid();

      const parsedFormData: ParsedChatInferenceDatapointRow = {
        dataset_name: datasetName,
        function_name: "write_haiku",
        name: "test_datapoint",
        id: originalId,
        episode_id: episodeId,
        input: {
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Write a haiku about coding",
                },
              ],
            },
          ],
        },
        output: [{ type: "text", text: "Code flows like water" }],
        tags: { environment: "test" },
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: sourceInferenceId,
        is_custom: false,
      };

      const { newId } = await saveDatapoint({
        parsedFormData,
        functionType: "chat",
      });

      // Verify new ID is different from original
      expect(newId).not.toBe(originalId);

      // Verify updateDatapoint was called with relevant customizations.
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          id: newId,
          function_name: "write_haiku",
          episode_id: undefined,
          // TODO: should assert on input and output
          tags: { environment: "test" },
          is_custom: true,
          source_inference_id: sourceInferenceId,
          auxiliary: "",
          name: "test_datapoint",
          staled_at: undefined,
        }),
      );

      // Verify deleteDatapoints was called with original ID
      expect(mockDeleteDatapoints).toHaveBeenCalledWith(datasetName, [
        originalId,
      ]);
    });
  });

  describe("saveDatapoint - json", () => {
    test("should generate new ID, call updateDatapoint with output_schema, and stale old datapoint", async () => {
      const datasetName = "test_dataset";
      const originalId = uuid();
      const sourceInferenceId = uuid();
      const episodeId = uuid();

      const parsedFormData: ParsedJsonInferenceDatapointRow = {
        dataset_name: datasetName,
        function_name: "extract_entities",
        name: "test_json_datapoint",
        id: originalId,
        episode_id: episodeId,
        input: {
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "John works at Google in Mountain View",
                },
              ],
            },
          ],
        },
        output: {
          raw: '{"person":["John"],"organization":["Google"],"location":["Mountain View"],"miscellaneous":[]}',
          parsed: {
            person: ["John"],
            organization: ["Google"],
            location: ["Mountain View"],
            miscellaneous: [],
          },
        },
        output_schema: {
          type: "object",
          properties: {
            person: { type: "array", items: { type: "string" } },
            organization: { type: "array", items: { type: "string" } },
            location: { type: "array", items: { type: "string" } },
            miscellaneous: { type: "array", items: { type: "string" } },
          },
          required: ["person", "organization", "location", "miscellaneous"],
        },
        tags: { source: "test" },
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: sourceInferenceId,
        is_custom: false,
      };

      const { newId } = await saveDatapoint({
        parsedFormData,
        functionType: "json",
      });

      // Verify new ID is different from original
      expect(newId).not.toBe(originalId);

      // Verify updateDatapoint was called with relevant customizations.
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          id: newId,
          episode_id: undefined,
          function_name: "extract_entities",
          tags: { source: "test" },
          // TODO: should assert on input and output.
          input: expect.any(Object),
          output: expect.any(Object),
          is_custom: true,
          source_inference_id: sourceInferenceId,
          output_schema: parsedFormData.output_schema,
          auxiliary: "",
          name: "test_json_datapoint",
          staled_at: undefined,
        }),
      );

      // Verify deleteDatapoints was called with original ID
      expect(mockDeleteDatapoints).toHaveBeenCalledWith(datasetName, [
        originalId,
      ]);
    });

    test("should handle json datapoint with null output", async () => {
      const parsedFormData: ParsedJsonInferenceDatapointRow = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        name: undefined,
        id: uuid(),
        episode_id: undefined,
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Test" }],
            },
          ],
        },
        output: undefined,
        output_schema: {
          type: "object",
          properties: {},
        },
        tags: {},
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: null,
        is_custom: false,
      };

      await saveDatapoint({
        parsedFormData,
        functionType: "json",
      });

      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.objectContaining({
          output: null,
        }),
      );
    });

    test("should handle mismatched function type by treating as chat datapoint", async () => {
      const parsedFormData: ParsedJsonInferenceDatapointRow = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        name: undefined,
        id: uuid(),
        episode_id: undefined,
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Test" }],
            },
          ],
        },
        output: undefined,
        output_schema: {
          type: "object",
          properties: {},
        },
        tags: {},
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: null,
        is_custom: false,
      };

      // When passing functionType="chat" with a JSON datapoint (which has output_schema),
      // the code treats it as a chat datapoint and loses the output_schema field
      const result = await saveDatapoint({
        parsedFormData,
        functionType: "chat",
      });

      expect(result.newId).toBeDefined();

      // Verify it was called without output_schema (treated as chat)
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.not.objectContaining({
          output_schema: expect.anything(),
        }),
      );
    });
  });

  describe("renameDatapoint - chat", () => {
    test("should update datapoint with new name", async () => {
      const datasetName = "test_dataset";
      const datapointId = uuid();
      const newName = "new_name";

      await renameDatapoint({
        datasetName,
        datapointId,
        name: newName,
      });

      // Verify updateDatapointsMetadata was called with the new name
      expect(mockUpdateDatapointsMetadata).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          datapoints: [
            {
              id: datapointId,
              metadata: {
                name: newName,
              },
            },
          ],
        }),
      );

      // Verify deleteDatapoints was NOT called (rename doesn't delete)
      expect(mockDeleteDatapoints).not.toHaveBeenCalled();
    });
  });

  describe("renameDatapoint - json", () => {
    test("should update datapoint with new name", async () => {
      const datasetName = "test_dataset";
      const datapointId = uuid();
      const newName = "new_json_name";

      await renameDatapoint({
        datasetName,
        datapointId,
        name: newName,
      });

      // Verify updateDatapointsMetadata was called with the new name
      expect(mockUpdateDatapointsMetadata).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          datapoints: [
            {
              id: datapointId,
              metadata: {
                name: newName,
              },
            },
          ],
        }),
      );

      // Verify deleteDatapoints was NOT called
      expect(mockDeleteDatapoints).not.toHaveBeenCalled();
    });
  });
});
