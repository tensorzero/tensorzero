import { describe, expect, test, beforeEach, vi } from "vitest";
import { v7 as uuid } from "uuid";
import {
  deleteDatapoint,
  saveDatapoint,
  renameDatapoint,
} from "./datapointOperations.server";
import type {
  DatasetCountInfo,
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "~/utils/clickhouse/datasets";
import type { Datapoint } from "~/utils/tensorzero";

// TODO(shuyangli): Once we remove all custom logic from the Node client, make mocking more ergonomic by providing a mock client at the tensorzero-node level.

// Mock TensorZero client at the module boundary
const mockUpdateDatapoint = vi.fn(
  async (_datasetName: string, datapoint: Datapoint) => ({
    id: datapoint.id,
  }),
);
vi.mock("~/utils/tensorzero.server", () => ({
  getTensorZeroClient: vi.fn(() => ({
    updateDatapoint: mockUpdateDatapoint,
  })),
}));

// Mock the datasets server functions
const mockGetDatasetCounts = vi.hoisted(() =>
  vi.fn<
    (arg: {
      function_name?: string;
      page_size?: number;
      offset?: number;
    }) => Promise<DatasetCountInfo[]>
  >(async () => []),
);
const mockStaleDatapoint = vi.hoisted(() =>
  vi.fn<
    (
      dataset_name: string,
      datapoint_id: string,
      function_type: "chat" | "json",
    ) => Promise<void>
  >(async () => {}),
);
vi.mock("~/utils/clickhouse/datasets.server", () => ({
  staleDatapoint: mockStaleDatapoint,
  getDatasetCounts: mockGetDatasetCounts,
}));

describe("datapointOperations", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
  });

  describe("deleteDatapoint", () => {
    test("should call staleDatapoint and redirect to /datasets when dataset is empty", async () => {
      // Mock getDatasetCounts to return empty array (no datasets)
      vi.mocked(mockGetDatasetCounts).mockResolvedValueOnce([]);

      const datasetName = "nonexistent_dataset";
      const datapointId = uuid();

      const result = await deleteDatapoint({
        dataset_name: datasetName,
        id: datapointId,
        functionType: "chat",
      });

      expect(mockStaleDatapoint).toHaveBeenCalledWith(
        datasetName,
        datapointId,
        "chat",
      );
      expect(result.redirectTo).toBe("/datasets");
    });

    test("should redirect to dataset page when dataset still has datapoints", async () => {
      const datasetName = "foo";
      const datapointId = uuid();

      // Mock mockGetDatasetCounts to return a dataset with count
      vi.mocked(mockGetDatasetCounts).mockResolvedValueOnce([
        {
          dataset_name: datasetName,
          count: 10,
          last_updated: "2025-04-15T02:33:58Z",
        },
      ]);

      const result = await deleteDatapoint({
        dataset_name: datasetName,
        id: datapointId,
        functionType: "json",
      });

      expect(mockStaleDatapoint).toHaveBeenCalledWith(
        datasetName,
        datapointId,
        "json",
      );
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
        tool_params: { temperature: 0.7 },
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
          episode_id: null,
          // TODO: should assert on input and output
          tags: { environment: "test" },
          is_custom: true,
          source_inference_id: sourceInferenceId,
          tool_params: { temperature: 0.7 },
          auxiliary: "",
          name: "test_datapoint",
          staled_at: null,
        }),
      );

      // Verify staleDatapoint was called with original ID
      expect(mockStaleDatapoint).toHaveBeenCalledWith(
        datasetName,
        originalId,
        "chat",
      );
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
          episode_id: null,
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
          staled_at: null,
        }),
      );

      // Verify mockStaleDatapoint was called with original ID
      expect(mockStaleDatapoint).toHaveBeenCalledWith(
        datasetName,
        originalId,
        "json",
      );
    });

    test("should handle json datapoint with null output", async () => {
      const parsedFormData: ParsedJsonInferenceDatapointRow = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        name: null,
        id: uuid(),
        episode_id: null,
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
        name: null,
        id: uuid(),
        episode_id: null,
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
      const sourceInferenceId = uuid();
      const episodeId = uuid();

      const datapoint: ParsedChatInferenceDatapointRow = {
        dataset_name: datasetName,
        function_name: "write_haiku",
        name: "old_name",
        id: datapointId,
        episode_id: episodeId,
        input: {
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "unstructured_text",
                  text: "Write a haiku about coding",
                },
              ],
            },
          ],
        },
        output: [{ type: "text", text: "Code flows like water" }],
        tool_params: { temperature: 0.7 },
        tags: { environment: "test" },
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: sourceInferenceId,
        is_custom: false,
      };

      const newName = "new_name";

      await renameDatapoint({
        functionType: "chat",
        datasetName,
        datapoint,
        newName,
      });

      // Verify updateDatapoint was called with the new name
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          id: datapointId,
          function_name: "write_haiku",
          episode_id: episodeId,
          tags: { environment: "test" },
          is_custom: false,
          source_inference_id: sourceInferenceId,
          tool_params: { temperature: 0.7 },
          auxiliary: "",
          name: newName,
          staled_at: null,
        }),
      );

      // Verify staleDatapoint was NOT called (rename doesn't stale)
      expect(mockStaleDatapoint).not.toHaveBeenCalled();
    });
  });

  describe("renameDatapoint - json", () => {
    test("should update datapoint with new name and preserve output_schema", async () => {
      const datasetName = "test_dataset";
      const datapointId = uuid();
      const sourceInferenceId = uuid();
      const episodeId = uuid();

      const datapoint: ParsedJsonInferenceDatapointRow = {
        dataset_name: datasetName,
        function_name: "extract_entities",
        name: "old_json_name",
        id: datapointId,
        episode_id: episodeId,
        input: {
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "unstructured_text",
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

      const newName = "new_json_name";

      await renameDatapoint({
        functionType: "json",
        datasetName,
        datapoint,
        newName,
      });

      // Verify updateDatapoint was called with the new name and output_schema
      expect(mockUpdateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          id: datapointId,
          episode_id: episodeId,
          function_name: "extract_entities",
          tags: { source: "test" },
          is_custom: false,
          source_inference_id: sourceInferenceId,
          output_schema: datapoint.output_schema,
          auxiliary: "",
          name: newName,
          staled_at: null,
        }),
      );

      // Verify staleDatapoint was NOT called
      expect(mockStaleDatapoint).not.toHaveBeenCalled();
    });

    test("should throw error when json datapoint is missing output_schema", async () => {
      const datapoint = {
        dataset_name: "test_dataset",
        function_name: "extract_entities",
        name: "old_name",
        id: uuid(),
        episode_id: null,
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "unstructured_text", text: "Test" }],
            },
          ],
        },
        output: undefined,
        tags: {},
        auxiliary: "",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: null,
        is_custom: false,
      } as ParsedJsonInferenceDatapointRow;

      await expect(
        renameDatapoint({
          functionType: "json",
          datasetName: "test_dataset",
          datapoint,
          newName: "new_name",
        }),
      ).rejects.toThrow("Json datapoint is missing output_schema");
    });
  });
});
