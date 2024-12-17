import { ParsedInferenceRow } from "../clickhouse";
import { SFTFormValues } from "~/routes/optimization.fine-tuning/route";

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

  abstract path_arg(): string | undefined;
  abstract query_params(): Record<string, string>;
  abstract display(): string;
  abstract result(): string | undefined;
  abstract provider(): string;
  abstract is_finished(): boolean;
  abstract poll(job: SFTJob): Promise<SFTJob>;
}
// TODO: unit test this thoroughly
export function format_url(base_url: string, job: SFTJob) {
  const queryString = new URLSearchParams(job.query_params()).toString();
  const formattedUrl = `${base_url}${
    job.path_arg() ? `/${job.path_arg()}` : ""
  }${queryString ? `?${queryString}` : ""}`;
  return formattedUrl;
}
