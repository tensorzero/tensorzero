import { ModelOption } from "./model_options";

export type SFTFormValues = {
  function: string;
  metric: string;
  model: ModelOption;
  variant: string;
  validationSplitPercent: number;
  maxSamples: number;
  threshold?: number;
};
