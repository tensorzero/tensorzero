import z from "zod";

export const VariantState = z.object({
  // Edited templates (if user has modified them)
  systemTemplate: z.string().optional(),
  assistantTemplate: z.string().optional(),
  userTemplate: z.string().optional(),
});

export type VariantState = z.infer<typeof VariantState>;

export const FunctionState = z.object({
  showSystemPrompt: z.boolean().default(false),
  showAssistantPrompt: z.boolean().default(false),
  showUserPrompt: z.boolean().default(false),
  showOutputSchema: z.boolean().default(false),
  showAdvanced: z.boolean().default(false),

  /** Order of variants to display */
  selectedVariants: z.array(z.string()).default([]),

  /** Edited templates for each variant */
  variants: z
    .record(z.string().describe("Variant name"), VariantState)
    .default({}),

  /** Selected dataset for this function */
  selectedDataset: z.string().optional(),
});

export type FunctionState = z.infer<typeof FunctionState>;

export const PlaygroundState = z
  .object({
    /** Most recently viewed function. Used as the default if there's no URL state. */
    lastViewedFunction: z.string().optional(),

    /** Selected variants and dataset for each function */
    functions: z
      .record(z.string().describe("Function name"), FunctionState)
      .default({}),
  })
  .default({});

export type PlaygroundState = z.infer<typeof PlaygroundState>;

export const DEFAULT_VARIANT_STATE: VariantState = {
  systemTemplate: undefined,
  assistantTemplate: undefined,
  userTemplate: undefined,
};

export const DEFAULT_FUNCTION_STATE: FunctionState = {
  selectedVariants: [],
  showSystemPrompt: false,
  showAssistantPrompt: false,
  showUserPrompt: false,
  showOutputSchema: false,
  showAdvanced: false,
  variants: {},
};

export const DEFAULT_PLAYGROUND_STATE: PlaygroundState = {
  functions: {},
};
