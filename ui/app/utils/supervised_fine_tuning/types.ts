import { OPENAI_ROLES } from "./constants";

export type OpenAIRole = (typeof OPENAI_ROLES)[number];

export type OpenAIMessage = {
  role: OpenAIRole;
  content?: string;
  tool_calls?: {
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }[];
  tool_call_id?: string;
  weight?: 0 | 1;
};

/**
 * Statistical distribution of numeric values
 */
export interface Distribution {
  min: number;
  max: number;
  mean: number;
  median: number;
  p5: number;
  p95: number;
}
