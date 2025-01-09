import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import { OpenAISFTJob } from "./openai";
import { FireworksSFTJob } from "./fireworks";
import type { SFTJob } from "./common";

export function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
  switch (data.model.provider) {
    case "openai":
      return OpenAISFTJob.fromFormData(data);
    case "fireworks":
      return FireworksSFTJob.from_form_data(data);
    default:
      throw new Error("Invalid provider");
  }
}
