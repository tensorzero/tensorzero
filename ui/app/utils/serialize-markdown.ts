import type {
  Input,
  StoredInference,
  InputMessageContent,
  ContentBlockChatOutput,
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
          : JSON.stringify(input.system, null, 2);
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
      return block.text ?? "";
    case "template":
      return JSON.stringify(block.arguments, null, 2);
    case "tool_call": {
      const args =
        "raw_arguments" in block ? block.raw_arguments : block.arguments;
      const name = "raw_name" in block ? block.raw_name : block.name;
      return `**Tool call: ${name}**\n\`\`\`json\n${typeof args === "string" ? args : JSON.stringify(args, null, 2)}\n\`\`\``;
    }
    case "tool_result":
      return `**Tool result: ${block.name}**\n\`\`\`\n${block.result}\n\`\`\``;
    case "file":
      return `[file: ${block.file_type}]`;
    case "unknown":
      return JSON.stringify(block.data, null, 2);
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

  // JSON inference output — wrap in code fence for readability
  if ("raw" in output) {
    const json = output.raw ?? JSON.stringify(output.parsed, null, 2);
    return `\`\`\`json\n${json}\n\`\`\``;
  }

  // Chat inference output (array of content blocks)
  return output.map(serializeChatOutputBlock).filter(Boolean).join("\n\n");
}

function serializeChatOutputBlock(block: ContentBlockChatOutput): string {
  switch (block.type) {
    case "text":
      return block.text;
    case "thought":
      return block.text ?? "";
    case "tool_call": {
      return `**Tool call: ${block.raw_name}**\n\`\`\`json\n${block.raw_arguments}\n\`\`\``;
    }
    case "unknown":
      return JSON.stringify(block.data, null, 2);
    default: {
      const _exhaustiveCheck: never = block;
      return _exhaustiveCheck;
    }
  }
}
