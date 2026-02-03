import type { JsonInferenceOutput } from "~/types/tensorzero";
import { logger } from "~/utils/logger";

/**
 * When we first introduced LLM Judges, we included the thinking section in the output.
 * We have since removed it, but we need to handle the old data.
 * So, we transform any old LLM Judge outputs to the new format by removing the thinking section from the
 * parsed and raw outputs.
 */
export function handle_llm_judge_output(output: string) {
  let parsed: JsonInferenceOutput;
  try {
    parsed = JSON.parse(output);
  } catch (e) {
    logger.warn("Error parsing LLM Judge output", e);
    // Don't do anything if the output failed to parse
    return output;
  }
  // biome-ignore lint/suspicious/noExplicitAny: legacy LLM Judge output has unknown shape
  if (!(parsed as any).parsed) {
    // if the output failed to parse don't do anything
    return output;
  }
  // biome-ignore lint/suspicious/noExplicitAny: legacy LLM Judge output has unknown shape
  if ((parsed as any).parsed.thinking) {
    // there is a thinking section that needs to be removed in the parsed and raw outputs
    // biome-ignore lint/suspicious/noExplicitAny: legacy LLM Judge output has unknown shape
    delete (parsed as any).parsed.thinking;
    const output = {
      // biome-ignore lint/suspicious/noExplicitAny: legacy LLM Judge output has unknown shape
      parsed: (parsed as any).parsed,
      // biome-ignore lint/suspicious/noExplicitAny: legacy LLM Judge output has unknown shape
      raw: JSON.stringify((parsed as any).parsed),
    };
    return JSON.stringify(output);
  }
  return JSON.stringify(parsed);
}
