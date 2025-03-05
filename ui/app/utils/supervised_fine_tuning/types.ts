export type OpenAIRole = "system" | "user" | "assistant" | "tool";

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
