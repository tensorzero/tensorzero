import type {
  ResolvedBase64Image,
  ResolvedInputMessageContent,
  StoragePath,
} from "./clickhouse/common";
import type { InputMessageContent } from "./clickhouse/common";
import type { ResolvedInputMessage } from "./clickhouse/common";
import type { InputMessage } from "./clickhouse/common";
import type { ResolvedInput } from "./clickhouse/common";
import type { Input } from "./clickhouse/common";
import { tensorZeroClient } from "./tensorzero.server";

export async function resolveInput(input: Input): Promise<ResolvedInput> {
  const resolvedMessages = await Promise.all(
    input.messages.map(async (message) => {
      return resolveMessage(message);
    }),
  );
  return {
    ...input,
    messages: resolvedMessages,
  };
}

async function resolveMessage(
  message: InputMessage,
): Promise<ResolvedInputMessage> {
  const resolvedContent = await Promise.all(
    message.content.map(async (content) => {
      return resolveContent(content);
    }),
  );
  return {
    ...message,
    content: resolvedContent,
  };
}

async function resolveContent(
  content: InputMessageContent,
): Promise<ResolvedInputMessageContent> {
  switch (content.type) {
    case "text":
    case "tool_call":
    case "tool_result":
    case "raw_text":
      return content;
    case "image":
      try {
        return {
          ...content,
          image: await resolveImage(content.storage_path),
        };
      } catch (error) {
        return {
          type: "image_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}

async function resolveImage(
  storagePath: StoragePath,
): Promise<ResolvedBase64Image> {
  const object = await tensorZeroClient.getObject(storagePath);
  return {
    url: object,
    mime_type: "image/png",
  };
}
