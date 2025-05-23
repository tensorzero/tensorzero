import type {
  FileContent,
  ModelInferenceInput,
  ModelInferenceInputMessage,
  ModelInferenceInputMessageContent,
  ResolvedBase64File,
  ResolvedInputMessageContent,
} from "./clickhouse/common";
import type { InputMessageContent } from "./clickhouse/common";
import type { ResolvedInputMessage } from "./clickhouse/common";
import type { InputMessage } from "./clickhouse/common";
import type { ResolvedInput } from "./clickhouse/common";
import type { Input } from "./clickhouse/common";
import { tensorZeroClient } from "./tensorzero.server";

export async function resolveInput(input: Input): Promise<ResolvedInput> {
  const resolvedMessages = await resolveMessages(input.messages);
  return {
    ...input,
    messages: resolvedMessages,
  };
}

export async function resolveModelInferenceInput(
  input: ModelInferenceInput,
): Promise<ResolvedInput> {
  const resolvedMessages = await resolveMessages(input.messages);
  return {
    ...input,
    messages: resolvedMessages,
  };
}

export async function resolveMessages(
  messages: InputMessage[],
): Promise<ResolvedInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveMessage(message);
    }),
  );
}

export async function resolveModelInferenceMessages(
  messages: ModelInferenceInputMessage[],
): Promise<ResolvedInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveModelInferenceMessage(message);
    }),
  );
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

async function resolveModelInferenceMessage(
  message: ModelInferenceInputMessage,
): Promise<ResolvedInputMessage> {
  const resolvedContent = await Promise.all(
    message.content.map(async (content) => {
      return resolveModelInferenceContent(content);
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
          type: "file",
          file: await resolveFile({
            type: "file",
            file: content.image,
            storage_path: content.storage_path
          }),
          storage_path: content.storage_path,
        };
      } catch (error) {
        return {
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }      
    case "file":
      try {
        return {
          ...content,
          file: await resolveFile(content as FileContent),
        };
      } catch (error) {
        return {
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}

async function resolveModelInferenceContent(
  content: ModelInferenceInputMessageContent,
): Promise<ResolvedInputMessageContent> {
  switch (content.type) {
    case "text":
      return {
        type: "text",
        value: content.text,
      };
    case "tool_call":
    case "tool_result":
    case "raw_text":
      return content;
    // Convert legacy 'image' content block to 'file' when resolving input
    case "image":
      try {
        return {
          type: "file",
          file: await resolveFile({
            type: "file",
            file: content.image,
            storage_path: content.storage_path
          }),
          storage_path: content.storage_path,
        };
      } catch (error) {
        return {
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
    case "file":
      try {
        return {
          ...content,
          file: await resolveFile(content as FileContent),
        };
      } catch (error) {
        return {
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}
async function resolveFile(
  content: FileContent,
): Promise<ResolvedBase64File> {
  const object = await tensorZeroClient.getObject(content.storage_path);
  const json = JSON.parse(object);
  const dataURL = `data:${content.file.mime_type};base64,${json.data}`;
  return {
    url: dataURL,
    mime_type: content.file.mime_type,
  };
}
