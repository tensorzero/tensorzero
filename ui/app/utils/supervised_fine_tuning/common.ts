import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type { ParsedInferenceExample } from "../clickhouse/curation";
import type { ProviderConfig } from "tensorzero-node";
import type { OptimizationJobHandle } from "tensorzero-node";

type ProviderType = ProviderConfig["type"];

export function splitValidationData(
  inferences: ParsedInferenceExample[],
  validationSplitPercent: number,
) {
  const validationSplit = validationSplitPercent / 100;
  const splitIndex =
    validationSplit > 0
      ? Math.floor(inferences.length * (1 - validationSplit))
      : inferences.length;

  const trainInferences = inferences.slice(0, splitIndex);
  const valInferences =
    validationSplitPercent > 0 ? inferences.slice(splitIndex) : [];

  return {
    trainInferences,
    valInferences,
  };
}

export type RawData =
  | {
      status: "ok";
      info: unknown;
    }
  | {
      status: "error";
      message: string;
    };

// export type SFTJobStatus = {"running" | "completed" | "error" | "idle";
export type SFTJobStatus =
  | {
      status: "running";
      modelProvider: ProviderType;
      formData: SFTFormValues;
      jobUrl: string;
      rawData: RawData;
      estimatedCompletionTime?: number;
    }
  | {
      status: "completed";
      modelProvider: ProviderType;
      formData: SFTFormValues;
      jobUrl: string;
      rawData: RawData;
      result: string;
    }
  | {
      status: "error";
      modelProvider: ProviderType;
      formData: SFTFormValues;
      jobUrl: string;
      rawData: RawData;
      error: string;
    }
  | { status: "idle" };

export abstract class SFTJob {
  protected constructor() {}

  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
  static fromFormData(data: SFTFormValues): Promise<SFTJob> {
    throw new Error(
      "Child class must implement fromFormData or from_job_handle",
    );
  }

  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
  static from_job_handle(jobHandle: OptimizationJobHandle): SFTJob {
    throw new Error(
      "Child class must implement fromFormData or from_job_handle",
    );
  }

  abstract status(): SFTJobStatus;
  abstract poll(): Promise<SFTJob>;
}
