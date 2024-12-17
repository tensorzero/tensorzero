import { SFTFormValues } from "~/routes/optimization.fine-tuning/route";

export async function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
  switch (data.model.provider) {
    case "openai": {
      const response = await fetch("/api/fine-tuning/openai", {
        method: "POST",
        body: JSON.stringify(data),
      });
      const json = await response.json();
      return new OpenAISFTJob(json.jobId, json.status, json.fineTunedModel);
    }
    case "fireworks": {
      const response = await fetch("/api/fine-tuning/fireworks", {
        method: "POST",
        body: JSON.stringify(data),
      });
      const json = await response.json();
      console.log("Fireworks job status at launch:", json);
      return new FireworksSFTJob(
        json.jobPath,
        json.status,
        json?.modelId,
        json?.modelPath,
      );
    }
    default:
      throw new Error(`Unsupported provider: ${data.model.provider}`);
  }
}

export async function poll_sft_job(jobStatus: SFTJob): Promise<SFTJob> {
  switch (jobStatus.provider()) {
    case "openai": {
      const url = format_url("/api/fine-tuning/openai", jobStatus);

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const json = await response.json();
      return new OpenAISFTJob(json.jobId, json.status, json.fineTunedModel);
    }
    case "fireworks": {
      const url = format_url("/api/fine-tuning/fireworks", jobStatus);
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
    default: {
      throw new Error(`Unsupported provider: ${jobStatus.provider}`);
    }
  }
}

export interface SFTJob {
  path_arg(): string | undefined;
  query_params(): Record<string, string>;
  display(): string; // TODO: we might want to make a more articulated thing to display, even a React component
  result(): string | undefined; // Returns the model name to use if the job is finished, otherwise undefined
  provider(): string;
  is_finished(): boolean;
}

// TODO: unit test this thoroughly
export function format_url(base_url: string, job: SFTJob) {
  console.log("foo Formatting URL", base_url, job);
  const queryString = new URLSearchParams(job.query_params()).toString();
  console.log("Query string:", queryString);
  const formattedUrl = `${base_url}${
    job.path_arg() ? `/${job.path_arg()}` : ""
  }${queryString ? `?${queryString}` : ""}`;
  console.log("Formatted URL:", formattedUrl);
  return formattedUrl;
}

export class OpenAISFTJob implements SFTJob {
  jobId: string;
  status: string;
  fineTunedModel: string | undefined;
  constructor(
    jobId: string,
    status: string,
    fineTunedModel: string | undefined,
  ) {
    this.jobId = jobId;
    this.status = status;
    this.fineTunedModel = fineTunedModel;
  }
  path_arg(): string | undefined {
    return this.jobId;
  }
  query_params(): Record<string, string> {
    return {};
  }
  display(): string {
    return this.status;
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
}

export class FireworksSFTJob implements SFTJob {
  jobPath: string;
  status: string;
  modelId: string | undefined;
  modelPath: string | undefined;
  constructor(
    jobPath: string,
    status: string,
    modelId: string | undefined,
    modelPath: string | undefined,
  ) {
    this.jobPath = jobPath;
    this.status = status;
    this.modelId = modelId;
    this.modelPath = modelPath;
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
    return this.status;
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
}
