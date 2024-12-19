import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
import { SFTJob } from "./common";

export class OpenAISFTJob extends SFTJob {
  constructor(
    public jobId: string,
    public status: string,
    public fineTunedModel?: string,
  ) {
    super();
  }

  static override async fromFormData(
    data: SFTFormValues,
  ): Promise<OpenAISFTJob> {
    const response = await fetch("/api/fine-tuning/openai", {
      method: "POST",
      body: JSON.stringify(data),
    });
    const json = await response.json();
    return new OpenAISFTJob(json.jobId, json.status, json.fineTunedModel);
  }

  path_arg(): string | undefined {
    return this.jobId;
  }
  query_params(): Record<string, string> {
    return {};
  }
  display(): string {
    return this.status ?? "";
  }
  result(): string | undefined {
    return this.fineTunedModel;
  }
  provider(): string {
    return "openai";
  }
  is_finished(): boolean {
    return this.status === "succeeded" || this.status === "failed";
  }

  async poll(): Promise<SFTJob> {
    const url = this.format_url("/api/fine-tuning/openai");

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    const json = await response.json();
    return new OpenAISFTJob(json.jobId, json.status, json.fineTunedModel);
  }
}
