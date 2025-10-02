import { describe, expect, test, beforeEach, vi } from "vitest";
import { v7 as uuid } from "uuid";
import {
  deleteDatapoint,
  saveDatapoint,
} from "./datapointOperations.server";
import type {
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "~/utils/clickhouse/datasets";

// Mock the TensorZero client
vi.mock("~/utils/tensorzero.server", () => {
  const mockUpdateDatapoint = vi.fn(async (datasetName: string, datapoint: any) => ({
    id: datapoint.id,
  }));

  return {
    getTensorZeroClient: vi.fn(() => ({
      updateDatapoint: mockUpdateDatapoint,
    })),
  };
});

// Mock the datasets server functions
vi.mock("~/utils/clickhouse/datasets.server", () => ({
  staleDatapoint: vi.fn(async () => {}),
  getDatasetCounts: vi.fn(async () => []),
}));

describe("datapointOperations", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
  });

  describe("deleteDatapoint", () => {
    test("should call staleDatapoint and redirect to /datasets when dataset is empty", async () => {
      const { getDatasetCounts, staleDatapoint } = await import(
        "~/utils/clickhouse/datasets.server"
      );

      // Mock getDatasetCounts to return empty array (no datasets)
      vi.mocked(getDatasetCounts).mockResolvedValueOnce([]);

      const datasetName = "nonexistent_dataset";
      const datapointId = uuid();

      const result = await deleteDatapoint({
        dataset_name: datasetName,
        id: datapointId,
        functionType: "chat",
      });

      expect(staleDatapoint).toHaveBeenCalledWith(
        datasetName,
        datapointId,
        "chat",
      );
      expect(result.redirectTo).toBe("/datasets");
    });

    test("should redirect to dataset page when dataset still has datapoints", async () => {
      const { getDatasetCounts, staleDatapoint } = await import(
        "~/utils/clickhouse/datasets.server"
      );

      const datasetName = "foo";
      const datapointId = uuid();

      // Mock getDatasetCounts to return a dataset with count
      vi.mocked(getDatasetCounts).mockResolvedValueOnce([
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

      expect(staleDatapoint).toHaveBeenCalledWith(
        datasetName,
        datapointId,
        "json",
      );
      expect(result.redirectTo).toBe(`/datasets/${datasetName}`);
    });
  });

  describe("saveDatapoint - chat", () => {
    test("should generate new ID, call updateDatapoint, and stale old datapoint", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");
      const { staleDatapoint } = await import("~/utils/clickhouse/datasets.server");

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
                { type: "unstructured_text", text: "Write a haiku about coding" },
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

      // Verify updateDatapoint was called
      const client = getTensorZeroClient();
      expect(client.updateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          function_name: "write_haiku",
          tags: { environment: "test" },
          is_custom: true,
          source_inference_id: sourceInferenceId,
          tool_params: { temperature: 0.7 },
        }),
      );

      // Verify staleDatapoint was called with original ID
      expect(staleDatapoint).toHaveBeenCalledWith(
        datasetName,
        originalId,
        "chat",
      );
    });

    test("should handle chat datapoint with null episode_id", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");

      const parsedFormData: ParsedChatInferenceDatapointRow = {
        dataset_name: "test_dataset",
        function_name: "write_haiku",
        name: null,
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
        output: [{ type: "text", text: "Output" }],
        tool_params: {},
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
        functionType: "chat",
      });

      const client = getTensorZeroClient();
      expect(client.updateDatapoint).toHaveBeenCalled();
    });

    test("should handle chat datapoint with auxiliary data", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");

      const parsedFormData: ParsedChatInferenceDatapointRow = {
        dataset_name: "test_dataset",
        function_name: "write_haiku",
        name: null,
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
        output: [{ type: "text", text: "Output" }],
        tool_params: {},
        tags: {},
        auxiliary: "some auxiliary data",
        is_deleted: false,
        updated_at: new Date().toISOString(),
        staled_at: null,
        source_inference_id: null,
        is_custom: false,
      };

      await saveDatapoint({
        parsedFormData,
        functionType: "chat",
      });

      const client = getTensorZeroClient();
      expect(client.updateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.objectContaining({
          auxiliary: "some auxiliary data",
        }),
      );
    });
  });

  describe("saveDatapoint - json", () => {
    test("should generate new ID, call updateDatapoint with output_schema, and stale old datapoint", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");
      const { staleDatapoint } = await import("~/utils/clickhouse/datasets.server");

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

      const { newId } = await saveDatapoint({
        parsedFormData,
        functionType: "json",
      });

      // Verify new ID is different from original
      expect(newId).not.toBe(originalId);

      // Verify updateDatapoint was called with output_schema
      const client = getTensorZeroClient();
      expect(client.updateDatapoint).toHaveBeenCalledWith(
        datasetName,
        expect.objectContaining({
          function_name: "extract_entities",
          tags: { source: "test" },
          is_custom: true,
          source_inference_id: sourceInferenceId,
          output_schema: parsedFormData.output_schema,
        }),
      );

      // Verify staleDatapoint was called with original ID
      expect(staleDatapoint).toHaveBeenCalledWith(
        datasetName,
        originalId,
        "json",
      );
    });

    test("should handle json datapoint with null output", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");

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
              content: [{ type: "unstructured_text", text: "Test" }],
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

      const client = getTensorZeroClient();
      expect(client.updateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.objectContaining({
          output: null,
        }),
      );
    });

    test("should handle mismatched function type by treating as chat datapoint", async () => {
      const { getTensorZeroClient } = await import("~/utils/tensorzero.server");
      const { staleDatapoint } = await import("~/utils/clickhouse/datasets.server");

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
              content: [{ type: "unstructured_text", text: "Test" }],
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
      const client = getTensorZeroClient();
      // Verify it was called without output_schema (treated as chat)
      expect(client.updateDatapoint).toHaveBeenCalledWith(
        "test_dataset",
        expect.not.objectContaining({
          output_schema: expect.anything(),
        }),
      );
    });
  });
});
