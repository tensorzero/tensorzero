import { describe, expect, test, beforeEach, vi } from "vitest";
import { v7 as uuid } from "uuid";
import {
  deleteDatapoint,
  updateDatapoint,
  renameDatapoint,
} from "./datapointOperations.server";
import type { UpdateDatapointFormData } from "./formDataUtils";
import type {
  GetDatasetMetadataParams,
  DatasetMetadata,
  UpdateDatapointsMetadataRequest,
  UpdateDatapointRequest,
} from "~/types/tensorzero";

// TODO(shuyangli): Once we remove all custom logic from the Node client, make mocking more ergonomic by providing a mock client at the tensorzero-node level.

// Mock TensorZero client at the module boundary
const mockUpdateDatapoint = vi.fn(
  async (_datasetName: string, _request: UpdateDatapointRequest) => ({
    id: uuid(), // Generate a new ID to simulate the backend creating a new datapoint
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

  describe("updateDatapoint - json", () => {
    test("should handle json datapoint with null output", async () => {
      const parsedFormData: Omit<UpdateDatapointFormData, "action"> = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        id: uuid(),
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Test" }],
            },
          ],
        },
        output: undefined,
        tags: {},
      };

      await updateDatapoint({
        parsedFormData,
        functionType: "json",
      });

      // When output is undefined, it should be omitted from the request
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.not.objectContaining({
          output: expect.anything(),
        }),
      );
    });

    test("should handle mismatched function type by treating as chat datapoint", async () => {
      const parsedFormData: Omit<UpdateDatapointFormData, "action"> = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        id: uuid(),
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Test" }],
            },
          ],
        },
        output: undefined,
        tags: {},
      };

      // When passing functionType="chat" with a JSON datapoint (which has output_schema),
      // the code treats it as a chat datapoint and loses the output_schema field
      const result = await updateDatapoint({
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
