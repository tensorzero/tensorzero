import type { ChatCompletionConfig } from "tensorzero-node";
import { create_env } from "../minijinja/pkg/minijinja_bindings";

export async function get_template_env(variant: ChatCompletionConfig) {
  const env: {
    system?: string;
    user?: string;
    assistant?: string;
  } = {};

  if ("system_template" in variant && variant.system_template) {
    env.system =
      typeof variant.system_template === "string"
        ? variant.system_template
        : (variant.system_template.contents ?? variant.system_template.path);
  }

  if ("user_template" in variant && variant.user_template) {
    env.user =
      typeof variant.user_template === "string"
        ? variant.user_template
        : (variant.user_template.contents ?? variant.user_template.path);
  }

  if ("assistant_template" in variant && variant.assistant_template) {
    env.assistant =
      typeof variant.assistant_template === "string"
        ? variant.assistant_template
        : (variant.assistant_template.contents ??
          variant.assistant_template.path);
  }

  return await create_env(env);
}
