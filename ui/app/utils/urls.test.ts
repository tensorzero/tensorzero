import { describe, it, expect } from "vitest";
import {
  toFunctionUrl,
  toVariantUrl,
  toInferenceUrl,
  toEpisodeUrl,
  toDatasetUrl,
  toDatapointUrl,
  toEvaluationUrl,
  toEvaluationDatapointUrl,
  toWorkflowEvaluationRunUrl,
  toWorkflowEvaluationProjectUrl,
} from "./urls";

describe("URL helper functions", () => {
  describe("toFunctionUrl", () => {
    it("should encode function names with special characters", () => {
      expect(toFunctionUrl("my_function")).toBe(
        "/observability/functions/my_function",
      );
      expect(toFunctionUrl("my_function/abc")).toBe(
        "/observability/functions/my_function%2Fabc",
      );
      expect(toFunctionUrl("func#test")).toBe(
        "/observability/functions/func%23test",
      );
      expect(toFunctionUrl("func?query")).toBe(
        "/observability/functions/func%3Fquery",
      );
    });
  });

  describe("toVariantUrl", () => {
    it("should encode both function and variant names", () => {
      expect(toVariantUrl("my_function", "variant_1")).toBe(
        "/observability/functions/my_function/variants/variant_1",
      );
      expect(toVariantUrl("func/abc", "var/xyz")).toBe(
        "/observability/functions/func%2Fabc/variants/var%2Fxyz",
      );
    });
  });

  describe("toInferenceUrl", () => {
    it("should encode inference IDs", () => {
      expect(toInferenceUrl("123")).toBe("/observability/inferences/123");
      expect(toInferenceUrl("id/with/slashes")).toBe(
        "/observability/inferences/id%2Fwith%2Fslashes",
      );
    });
  });

  describe("toEpisodeUrl", () => {
    it("should encode episode IDs", () => {
      expect(toEpisodeUrl("456")).toBe("/observability/episodes/456");
      expect(toEpisodeUrl("ep#123")).toBe("/observability/episodes/ep%23123");
    });
  });

  describe("toDatasetUrl", () => {
    it("should encode dataset names", () => {
      expect(toDatasetUrl("my_dataset")).toBe("/datasets/my_dataset");
      expect(toDatasetUrl("dataset/test")).toBe("/datasets/dataset%2Ftest");
    });
  });

  describe("toDatapointUrl", () => {
    it("should encode both dataset name and datapoint ID", () => {
      expect(toDatapointUrl("my_dataset", "point_1")).toBe(
        "/datasets/my_dataset/datapoint/point_1",
      );
      expect(toDatapointUrl("dataset/test", "point#1")).toBe(
        "/datasets/dataset%2Ftest/datapoint/point%231",
      );
    });
  });

  describe("toEvaluationUrl", () => {
    it("should encode evaluation names", () => {
      expect(toEvaluationUrl("eval_1")).toBe("/evaluations/eval_1");
      expect(toEvaluationUrl("eval/test")).toBe("/evaluations/eval%2Ftest");
    });

    it("should handle query parameters", () => {
      expect(
        toEvaluationUrl("eval_1", { evaluation_run_ids: "run_1,run_2" }),
      ).toBe("/evaluations/eval_1?evaluation_run_ids=run_1%2Crun_2");
    });
  });

  describe("toEvaluationDatapointUrl", () => {
    it("should encode evaluation name and datapoint ID", () => {
      expect(toEvaluationDatapointUrl("eval_1", "point_1")).toBe(
        "/evaluations/eval_1/point_1",
      );
      expect(toEvaluationDatapointUrl("eval/test", "point#1")).toBe(
        "/evaluations/eval%2Ftest/point%231",
      );
    });

    it("should handle query parameters", () => {
      expect(
        toEvaluationDatapointUrl("eval_1", "point_1", {
          evaluation_run_ids: "run_1",
        }),
      ).toBe("/evaluations/eval_1/point_1?evaluation_run_ids=run_1");
    });
  });

  describe("toWorkflowEvaluationRunUrl", () => {
    it("should encode run IDs", () => {
      expect(toWorkflowEvaluationRunUrl("run_1")).toBe(
        "/dynamic_evaluations/runs/run_1",
      );
      expect(toWorkflowEvaluationRunUrl("run/test")).toBe(
        "/dynamic_evaluations/runs/run%2Ftest",
      );
    });
  });

  describe("toWorkflowEvaluationProjectUrl", () => {
    it("should encode project names", () => {
      expect(toWorkflowEvaluationProjectUrl("project_1")).toBe(
        "/dynamic_evaluations/projects/project_1",
      );
      expect(toWorkflowEvaluationProjectUrl("project/test")).toBe(
        "/dynamic_evaluations/projects/project%2Ftest",
      );
    });
  });
});
