import type { InputMessageContent, Role } from "../clickhouse/common";
import { JsExposedEnv } from "../minijinja/pkg/minijinja_bindings";

/**
 * Renders a text message using the provided environment and content.
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
  content: InputMessageContent,
) {
  if (content.type !== "text") {
    throw new Error("Content must be a text block");
  }
  if (typeof content.value === "string") {
    return content.value;
  }
  return env.render(role, content.value);
}
