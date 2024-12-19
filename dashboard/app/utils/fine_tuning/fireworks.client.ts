import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
import { SFTJob } from "./common";

export class FireworksSFTJob extends SFTJob {
  constructor(
    public jobPath: string,
    public status: string,
    public modelId?: string,
    public modelPath?: string,
  ) {
    super();
  }

  static override async fromFormData(
    data: SFTFormValues,
  ): Promise<FireworksSFTJob> {
    const response = await fetch("/api/fine-tuning/fireworks", {
      method: "POST",
      body: JSON.stringify(data),
    });
    const json = await response.json();
    return new FireworksSFTJob(
      json.jobPath,
      json.status,
      json?.modelId,
      json?.modelPath,
    );
  }

  path_arg(): string | undefined {
    return undefined;
  }

  query_params(): Record<string, string> {
    const params: Record<string, string> = { jobPath: this.jobPath };
    if (this.modelId) {
      params.modelId = this.modelId;
    }
    return params;
  }

  display(): string {
    return this.status ?? "";
  }

  result(): string | undefined {
    return this.modelPath;
  }

  provider(): string {
    return "fireworks";
  }

  is_finished(): boolean {
    return this.status === "deployed" || this.status === "failed";
  }

  async poll(): Promise<SFTJob> {
    const url = this.format_url("/api/fine-tuning/fireworks");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    const json = await response.json();
    return new FireworksSFTJob(
      json.jobPath,
      json.status,
      json?.modelId,
      json?.modelPath,
    );
  }
}
