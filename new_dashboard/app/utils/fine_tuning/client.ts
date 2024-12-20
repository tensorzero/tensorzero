import type { SFTFormValues } from "~/routes/optimization/fine-tuning/types";
//import { OpenAISFTJob } from "./openai.client";
import { FireworksSFTJob } from "./fireworks";
// import { SFTJob } from "./common";

export function launch_sft_job(data: SFTFormValues): Promise<FireworksSFTJob> {
  switch (data.model.provider) {
    // case "openai":
    //   console.log("OpenAI client:", OpenAISFTJob);
    //   return OpenAISFTJob.fromFormData(data);
    case "fireworks":
      console.log("Fireworks client:", FireworksSFTJob);
      return FireworksSFTJob.from_form_data(data);
    default:
      throw new Error("Invalid provider");
  }
}
