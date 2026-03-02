import type {
  Input,
  StoredInference,
  InputMessageContent,
  ContentBlockChatOutput,
  JsonValue,
} from "~/types/tensorzero";

/**
 * Serializes an inference's input + output as a markdown conversation.
 *
 * Format:
 * ## system
 * <system prompt text>
 *
 * ## user
 * <user message>
 *
 * ## assistant
 * <assistant output>
 */
export function serializeConversationMarkdown(
  input: Input | undefined,
  output: StoredInference["output"],
): string {
  const sections: string[] = [];

  if (input) {
    // System prompt
    if (input.system !== undefined) {
      const systemText =
        typeof input.system === "string"
          ? input.system
          : serializeJsonValue(input.system);
      sections.push(`## system\n\n${systemText}`);
    }

    // Input messages
    for (const message of input.messages) {
      const contentText = message.content
        .map(serializeInputContent)
        .filter(Boolean)
        .join("\n\n");
      sections.push(`## ${message.role}\n\n${contentText}`);
    }
  }

  // Output
  if (output !== undefined) {
    sections.push(`## assistant\n\n${serializeOutput(output)}`);
  }

  return sections.join("\n\n");
}

function serializeInputContent(block: InputMessageContent): string {
  switch (block.type) {
    case "text":
      return block.text;
    case "raw_text":
      return block.value;
    case "thought":
      return `*${block.text ?? ""}*`;
    case "template":
      return serializeJsonValue(block.arguments);
    case "tool_call": {
      const args =
        "raw_arguments" in block ? block.raw_arguments : block.arguments;
      const name = "raw_name" in block ? block.raw_name : block.name;
      const argsText =
        typeof args === "string"
          ? tryParseAndSerialize(args)
          : serializeJsonValue(args);
      return `**Tool call: ${name}**\n\n${argsText}`;
    }
    case "tool_result":
      return `**Tool result: ${block.name}**\n\n${block.result}`;
    case "file":
      return `[file: ${block.file_type}]`;
    case "unknown":
      return serializeJsonValue(block.data);
    default: {
      const _exhaustiveCheck: never = block;
      return _exhaustiveCheck;
    }
  }
}

function serializeOutput(output: StoredInference["output"]): string {
  if (output === undefined) {
    return "";
  }

  // JSON inference output — parse and render as markdown
  if ("raw" in output) {
    const raw = output.raw;
    if (raw !== null && raw !== undefined) {
      return tryParseAndSerialize(raw);
    }
    if (output.parsed !== null && output.parsed !== undefined) {
      return serializeJsonValue(output.parsed);
    }
    return "";
  }

  // Chat inference output (array of content blocks)
  return output.map(serializeChatOutputBlock).filter(Boolean).join("\n\n");
}

function serializeChatOutputBlock(block: ContentBlockChatOutput): string {
  switch (block.type) {
    case "text":
      return block.text;
    case "thought":
      return `*${block.text ?? ""}*`;
    case "tool_call": {
      const argsText = tryParseAndSerialize(block.raw_arguments);
      return `**Tool call: ${block.raw_name}**\n\n${argsText}`;
    }
    case "unknown":
      return serializeJsonValue(block.data);
    default: {
      const _exhaustiveCheck: never = block;
      return _exhaustiveCheck;
    }
  }
}

/**
 * Try to parse a JSON string and render as markdown key-value pairs.
 * Falls back to the raw string if it's not valid JSON or not an object.
 */
function tryParseAndSerialize(raw: string): string {
  try {
    const parsed: unknown = JSON.parse(raw);
    return serializeJsonValue(parsed as JsonValue);
  } catch {
    return raw;
  }
}

/**
 * Recursively renders a JSON value as readable markdown.
 * - Objects → **key**: value lines (nested objects indent)
 * - Arrays → bulleted lists
 * - Scalars → plain text
 */
function serializeJsonValue(value: JsonValue, depth: number = 0): string {
  if (value === null || value === undefined) {
    return "null";
  }

  if (typeof value === "string") {
    // If the string looks like JSON (object or array), try to parse and recurse
    const trimmed = value.trimStart();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const parsed: unknown = JSON.parse(value);
        if (typeof parsed === "object" && parsed !== null) {
          return serializeJsonValue(parsed as JsonValue, depth);
        }
      } catch {
        // Not valid JSON — fall through to return raw string
      }
    }
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return "*(empty)*";
    return value
      .map((item) => `- ${serializeJsonValue(item, depth + 1)}`)
      .join("\n");
  }

  // Object
  const entries = Object.entries(value);
  if (entries.length === 0) return "*(empty)*";

  return entries
    .map(([key, val]) => {
      const serialized = serializeJsonValue(val, depth + 1);
      // If the value is multi-line (nested object/array), put it on the next line
      if (serialized.includes("\n")) {
        return `**${key}**:\n${serialized}`;
      }
      return `**${key}**: ${serialized}`;
    })
    .join("\n");
}
