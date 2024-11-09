import { InputMessageContent, Role } from "utils/clickhouse";
import { JsExposedEnv } from "../minijinja/pkg";

/**
 * Renders a message using the provided environment and content.
 *
 * @param env - The Minijinja environment for rendering (should contain templates for "system", "user" and "assistant" if and only
 *                                                       if the function takes structured input)
 * @param role - The role of the message sender (must be "user" or "assistant")
 * @param content - Array of message content blocks (must contain exactly one text block)
 * @returns The rendered message string
 * @throws {Error} If role is invalid, content length is not 1, or content type is not text
 */
export function render_message(
  env: JsExposedEnv,
  role: Role,
  content: InputMessageContent[],
) {
  if (role !== "user" && role !== "assistant") {
    throw new Error('Role must be either "user" or "assistant"');
  }
  if (content.length !== 1) {
    throw new Error("Content must contain exactly one block");
  }
  if (content[0].type !== "text") {
    throw new Error("Content must be a text block");
  }
  if (typeof content[0].value === "string") {
    return content[0].value;
  }
  return env.render(role, content[0].value);
}
