import { z } from "zod";

export const EvaluationErrorSchema = z.object({
  datapoint_id: z.string(),
  message: z.string(),
});
export type EvaluationError = z.infer<typeof EvaluationErrorSchema>;

export interface DisplayEvaluationError {
  datapoint_id?: string;
  message: string;
}
