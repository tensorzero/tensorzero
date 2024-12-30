import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import type { ParsedInferenceRow } from "../clickhouse";
import type { ProviderType } from "../config/models";
import type { ProgressInfo } from "~/routes/optimization/supervised-fine-tuning/ProgressIndicator";

export function splitValidationData(
  inferences: ParsedInferenceRow[],
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

export type SFTJobStatus = "running" | "completed" | "error" | "idle";

// Abstract base class
export abstract class SFTJob {
  protected constructor() {}

  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
  static fromFormData(data: SFTFormValues): Promise<SFTJob> {
    throw new Error("Child class must implement fromFormData");
  }

  abstract result(): string | undefined;
  abstract status(): SFTJobStatus;
  abstract progress_info(): ProgressInfo;
  abstract poll(): Promise<SFTJob>;
  abstract provider(): ProviderType;
}
