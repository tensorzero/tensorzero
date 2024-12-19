import { SFTFormValues } from "~/routes/optimization.fine-tuning/types";
import { OpenAISFTJob } from "./openai.client";
import { FireworksSFTJob } from "./fireworks.client";
import { SFTJob } from "./common";

export function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
  switch (data.model.provider) {
    case "openai":
      return OpenAISFTJob.fromFormData(data);
    case "fireworks":
      return FireworksSFTJob.fromFormData(data);
    default:
      throw new Error("Invalid provider");
  }
}
