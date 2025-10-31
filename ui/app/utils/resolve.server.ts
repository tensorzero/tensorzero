import type {
  DisplayInput,
  DisplayInputMessage,
  DisplayInputMessageContent,
  FileContent,
  Input,
  InputMessage,
  InputMessageContent,
  ModelInferenceInputMessage,
  ModelInferenceInputMessageContent,
  ResolvedBase64File,
  Role,
  LegacyTextInput,
} from "./clickhouse/common";
import type {
  FunctionConfig,
  JsonValue,
  StoredInput,
  StoredInputMessage,
  StoredInputMessageContent,
} from "tensorzero-node";
import { getTensorZeroClient } from "./tensorzero.server";

export async function resolveInput(
  input: Input,
  functionConfig: FunctionConfig | null,
): Promise<DisplayInput> {
  const resolvedMessages = await resolveMessages(
    input.messages,
    functionConfig,
  );
  return {
    ...input,
    messages: resolvedMessages,
  };
}

export async function resolveMessages(
  messages: InputMessage[],
  functionConfig: FunctionConfig | null,
): Promise<DisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveMessage(message, functionConfig);
    }),
  );
}

export async function resolveModelInferenceMessages(
  messages: ModelInferenceInputMessage[],
): Promise<DisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveModelInferenceMessage(message);
    }),
  );
}
async function resolveMessage(
  message: InputMessage,
  functionConfig: FunctionConfig | null,
): Promise<DisplayInputMessage> {
  const resolvedContent = await Promise.all(
    message.content.map(async (content) => {
      return resolveContent(content, message.role, functionConfig);
    }),
  );
  return {
    ...message,
    content: resolvedContent,
  };
}

async function resolveModelInferenceMessage(
  message: ModelInferenceInputMessage,
): Promise<DisplayInputMessage> {
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
  role: Role,
  functionConfig: FunctionConfig | null,
): Promise<DisplayInputMessageContent> {
  switch (content.type) {
    case "tool_call":
    case "tool_result":
    case "raw_text":
    case "thought":
    case "unknown":
    case "template":
      return content;
    case "text":
      return prepareDisplayText(content, role, functionConfig);
    case "image":
      try {
        return {
          type: "file",
          file: await resolveFile({
            type: "file",
            file: content.image,
            storage_path: content.storage_path,
          }),
          storage_path: content.storage_path,
        };
      } catch (error) {
        return {
          file: {
            url: content.image.url,
            mime_type: content.image.mime_type,
          },
          storage_path: content.storage_path,
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
    case "file":
      try {
        return {
          ...content,
          file: await resolveFile(content),
        };
      } catch (error) {
        return {
          ...content,
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}

async function resolveModelInferenceContent(
  content: ModelInferenceInputMessageContent,
): Promise<DisplayInputMessageContent> {
  switch (content.type) {
    case "text":
      // Do not use prepareDisplayText here because these are model inferences and should be post-templating
      // and will always be unstructured text.
      return {
        type: "text",
        text: content.text,
      };
    case "tool_call":
    case "tool_result":
    case "raw_text":
    case "thought":
    case "unknown":
      return content;
    // Convert legacy 'image' content block to 'file' when resolving input
    case "image":
      try {
        return {
          type: "file",
          file: await resolveFile({
            type: "file",
            file: content.image,
            storage_path: content.storage_path,
          }),
          storage_path: content.storage_path,
        };
      } catch (error) {
        return {
          file: {
            url: null,
            mime_type: content.image.mime_type,
          },
          storage_path: content.storage_path,
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
    case "file":
      try {
        return {
          ...content,
          file: await resolveFile(content),
        };
      } catch (error) {
        return {
          ...content,
          type: "file_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}
async function resolveFile(content: FileContent): Promise<ResolvedBase64File> {
  const object = await getTensorZeroClient().getObject(content.storage_path);
  const json = JSON.parse(object);
  const data = `data:${content.file.mime_type};base64,${json.data}`;
  return {
    data,
    mime_type: content.file.mime_type,
  };
}

// In the current data model we can't distinguish between a message being a structured one from a schema
// or an unstructured one without a schema without knowing the function config.
// So as we prepare the input for display, we check this and return an unambiguous type of structured or unstructured text.
// TODO (Gabriel): this function uses legacy types and should be deprecated ASAP. It won't handle sad paths very well.
function prepareDisplayText(
  textBlock: LegacyTextInput,
  role: Role,
  functionConfig: FunctionConfig | null,
): DisplayInputMessageContent {
  // If there's no function config, we can't do any templating because of legacy templates...
  if (!functionConfig) {
    return {
      type: "missing_function_text",
      value:
        typeof textBlock.value === "string"
          ? textBlock.value
          : JSON.stringify(textBlock.value),
    };
  }

  if (textBlock.text !== undefined) {
    return {
      type: "text",
      text: textBlock.text,
    };
  }

  // Handle the legacy structured prompts that were stored as text content blocks
  if (role === "user" && functionConfig.schemas["user"] !== undefined) {
    return {
      type: "template",
      name: "user",
      arguments: (() => {
        if (
          typeof textBlock.value === "object" &&
          textBlock.value !== null &&
          !Array.isArray(textBlock.value)
        ) {
          return textBlock.value as Record<string, JsonValue>;
        }
        throw new Error(
          `Invalid arguments for user template: expected object, got ${typeof textBlock.value}`,
        );
      })(),
    };
  }

  if (
    role === "assistant" &&
    functionConfig.schemas["assistant"] !== undefined
  ) {
    return {
      type: "template",
      name: "assistant",
      arguments: (() => {
        if (
          typeof textBlock.value === "object" &&
          textBlock.value !== null &&
          !Array.isArray(textBlock.value)
        ) {
          return textBlock.value as Record<string, JsonValue>;
        }
        throw new Error(
          `Invalid arguments for assistant template: expected object, got ${typeof textBlock.value}`,
        );
      })(),
    };
  }

  // Otherwise it's just unstructured text
  return {
    type: "text",
    text:
      typeof textBlock.value === "string"
        ? textBlock.value
        : JSON.stringify(textBlock.value),
  };
}

// ===== StoredInput =====
// TODO: These functions should be deprecated as we clean up the types...

export async function resolveStoredInput(
  input: StoredInput,
): Promise<DisplayInput> {
  const resolvedMessages = await resolveStoredInputMessages(input.messages);
  return {
    ...input,
    messages: resolvedMessages,
  };
}

export async function resolveStoredInputMessages(
  messages: StoredInputMessage[],
): Promise<DisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveStoredInputMessage(message);
    }),
  );
}

async function resolveStoredInputMessage(
  message: StoredInputMessage,
): Promise<DisplayInputMessage> {
  const resolvedContent = await Promise.all(
    message.content.map(async (content) => {
      return resolveStoredInputMessageContent(content);
    }),
  );
  return {
    ...message,
    content: resolvedContent,
  };
}

async function resolveStoredInputMessageContent(
  content: StoredInputMessageContent,
): Promise<DisplayInputMessageContent> {
  switch (content.type) {
    case "tool_call":
    case "tool_result":
    case "raw_text":
    case "thought":
    case "unknown":
    case "template":
    case "text":
      return content;
    case "file":
      try {
        // Convert flattened ObjectStorageFile to nested FileContent structure
        const fileContent: FileContent = {
          type: "file",
          file: {
            url: content.source_url ?? null,
            mime_type: content.mime_type,
          },
          storage_path: content.storage_path,
        };
        const resolvedFile = await resolveFile(fileContent);
        return {
          type: "file",
          file: {
            data: resolvedFile.data,
            mime_type: resolvedFile.mime_type,
          },
          storage_path: content.storage_path,
        };
      } catch (error) {
        return {
          type: "file_error",
          file: {
            url: content.source_url ?? null,
            mime_type: content.mime_type,
          },
          storage_path: content.storage_path,
          error: error instanceof Error ? error.message : String(error),
        };
      }
  }
}
