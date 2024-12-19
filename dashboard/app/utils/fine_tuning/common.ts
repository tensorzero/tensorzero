console.log("Loading common.ts");
import { ParsedInferenceRow } from "../clickhouse";
import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";

export function splitValidationData(
  inferences: ParsedInferenceRow[],
  validationSplit: number,
) {
  const splitIndex =
    validationSplit > 0
      ? Math.floor(inferences.length * (1 - validationSplit))
      : inferences.length;

  const trainInferences = inferences.slice(0, splitIndex);
  const valInferences = validationSplit > 0 ? inferences.slice(splitIndex) : [];

  return {
    trainInferences,
    valInferences,
  };
}

// Abstract base class
export abstract class SFTJob {
  protected constructor() {}

  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
  static fromFormData(data: SFTFormValues): Promise<SFTJob> {
    throw new Error("Child class must implement fromFormData");
  }

  format_url(base_url: string): string {
    const queryString = new URLSearchParams(this.query_params()).toString();
    const formattedUrl = `${base_url}${
      this.path_arg() ? `/${this.path_arg()}` : ""
    }${queryString ? `?${queryString}` : ""}`;
    return formattedUrl;
  }

  abstract path_arg(): string | undefined;
  abstract query_params(): Record<string, string>;
  abstract display(): string;
  abstract result(): string | undefined;
  abstract provider(): string;
  abstract is_finished(): boolean;
  abstract poll(): Promise<SFTJob>;
}
