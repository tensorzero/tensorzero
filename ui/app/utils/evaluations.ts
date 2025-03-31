import { z } from "zod";

export const EvalErrorSchema = z.object({
  datapoint_id: z.string(),
  message: z.string(),
});
export type EvalError = z.infer<typeof EvalErrorSchema>;

export interface DisplayEvalError {
  datapoint_id?: string;
  message: string;
}
