import { OPENAI_ROLES } from "./constants";

export type OpenAIRole = (typeof OPENAI_ROLES)[number];

export type OpenAIMessage = {
  role: OpenAIRole;
  content?: string | Array<{
    type: "image_url";
    image_url: {
      url: string;
    };
  }>;
  name?: string;
  function_call?: {
    name: string;
    arguments: string;
  };
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

/**
 * Function parameter types for token calculation
 */
export interface FunctionParameter {
  type: string;
  description: string;
  enum?: string[];
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: {
    type: string;
    properties: Record<string, FunctionParameter>;
  };
}

export interface ToolFunction {
  type: string;
  function: FunctionDefinition;
}
