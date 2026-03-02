import type {
  ContentBlockChatOutput,
  Input,
  InputMessage,
  InputMessageContent,
  StoredInference,
  System,
} from "~/types/tensorzero";

interface SerializedMessage {
  role: "system" | "user" | "assistant";
  content: string | SerializedContentBlock[];
}

type SerializedContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: string }
  | { type: "tool_result"; id: string; name: string; result: string }
  | { type: "thought"; text: string }
  | { type: "file"; file_type: string }
  | { type: "unknown"; data: unknown };

function serializeSystem(system: System): string {
  if (typeof system === "string") {
    return system;
  }
  return JSON.stringify(system);
}

function serializeInputContentBlock(
  block: InputMessageContent,
): SerializedContentBlock {
  switch (block.type) {
    case "text":
      return { type: "text", text: block.text };
    case "raw_text":
      return { type: "text", text: block.value };
    case "template":
      return { type: "text", text: JSON.stringify(block.arguments) };
    case "tool_call": {
      // ToolCallWrapper = ToolCall | InferenceResponseToolCall
      // InferenceResponseToolCall has `raw_arguments`, ToolCall has `arguments: string`
      const args =
        "raw_arguments" in block ? block.raw_arguments : block.arguments;
      const name =
        "raw_name" in block ? (block.name ?? block.raw_name) : block.name;
      return { type: "tool_call", id: block.id, name, arguments: args };
    }
    case "tool_result":
      return {
        type: "tool_result",
        id: block.id,
        name: block.name,
        result: block.result,
      };
    case "thought":
      return { type: "thought", text: block.text ?? "" };
    case "file":
      return { type: "file", file_type: block.file_type };
    case "unknown":
      return { type: "unknown", data: block.data };
    default: {
      const _exhaustiveCheck: never = block;
      return _exhaustiveCheck;
    }
  }
}

function serializeOutputContentBlock(
  block: ContentBlockChatOutput,
): SerializedContentBlock {
  switch (block.type) {
    case "text":
      return { type: "text", text: block.text };
    case "tool_call":
      return {
        type: "tool_call",
        id: block.id,
        name: block.name ?? block.raw_name,
        arguments: block.raw_arguments,
      };
    case "thought":
      return { type: "thought", text: block.text ?? "" };
    case "unknown":
      return { type: "unknown", data: block.data };
    default: {
      const _exhaustiveCheck: never = block;
      return _exhaustiveCheck;
    }
  }
}

function serializeInputMessage(message: InputMessage): SerializedMessage {
  const blocks = message.content.map(serializeInputContentBlock);
  return {
    role: message.role,
    content: maybeFlattenContent(blocks),
  };
}

/**
 * If the content is a single text block, flatten to a plain string.
 */
function maybeFlattenContent(
  blocks: SerializedContentBlock[],
): string | SerializedContentBlock[] {
  if (blocks.length === 1 && blocks[0].type === "text") {
    return blocks[0].text;
  }
  return blocks;
}

export function serializeConversation(
  input: Input | undefined,
  inference: StoredInference,
): string {
  const messages: SerializedMessage[] = [];

  // System prompt
  if (input?.system != null) {
    messages.push({
      role: "system",
      content: serializeSystem(input.system),
    });
  }

  // Input messages
  if (input?.messages) {
    for (const msg of input.messages) {
      messages.push(serializeInputMessage(msg));
    }
  }

  // Output (assistant turn)
  if (inference.type === "chat" && inference.output) {
    const blocks = inference.output.map(serializeOutputContentBlock);
    messages.push({
      role: "assistant",
      content: maybeFlattenContent(blocks),
    });
  } else if (inference.type === "json" && inference.output) {
    const content =
      inference.output.raw ?? JSON.stringify(inference.output.parsed);
    messages.push({
      role: "assistant",
      content,
    });
  }

  return JSON.stringify(messages, null, 2);
}
