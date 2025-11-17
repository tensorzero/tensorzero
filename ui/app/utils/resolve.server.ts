import type {
  ZodDisplayInput,
  ZodDisplayInputMessage,
  ZodDisplayInputMessageContent,
  ZodFileContent,
  ZodInput,
  ZodInputMessage,
  ZodInputMessageContent,
  ZodModelInferenceInputMessage,
  ZodModelInferenceInputMessageContent,
  ZodResolvedBase64File,
  ZodRole,
  ZodLegacyTextInput,
} from "./clickhouse/common";
import type {
  FunctionConfig,
  JsonValue,
  StoredInput,
  StoredInputMessage,
  StoredInputMessageContent,
  Input,
  InputMessageContent,
} from "~/types/tensorzero";
import { getTensorZeroClient } from "./tensorzero.server";

export async function resolveInput(
  input: ZodInput,
  functionConfig: FunctionConfig | null,
): Promise<ZodDisplayInput> {
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
  messages: ZodInputMessage[],
  functionConfig: FunctionConfig | null,
): Promise<ZodDisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveMessage(message, functionConfig);
    }),
  );
}

export async function resolveModelInferenceMessages(
  messages: ZodModelInferenceInputMessage[],
): Promise<ZodDisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveModelInferenceMessage(message);
    }),
  );
}
async function resolveMessage(
  message: ZodInputMessage,
  functionConfig: FunctionConfig | null,
): Promise<ZodDisplayInputMessage> {
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
  message: ZodModelInferenceInputMessage,
): Promise<ZodDisplayInputMessage> {
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
  content: ZodInputMessageContent,
  role: ZodRole,
  functionConfig: FunctionConfig | null,
): Promise<ZodDisplayInputMessageContent> {
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
  content: ZodModelInferenceInputMessageContent,
): Promise<ZodDisplayInputMessageContent> {
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
async function resolveFile(
  content: ZodFileContent,
): Promise<ZodResolvedBase64File> {
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
  textBlock: ZodLegacyTextInput,
  role: ZodRole,
  functionConfig: FunctionConfig | null,
): ZodDisplayInputMessageContent {
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
): Promise<ZodDisplayInput> {
  const resolvedMessages = await resolveStoredInputMessages(input.messages);
  return {
    ...input,
    messages: resolvedMessages,
  };
}

export async function resolveStoredInputMessages(
  messages: StoredInputMessage[],
): Promise<ZodDisplayInputMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      return resolveStoredInputMessage(message);
    }),
  );
}

async function resolveStoredInputMessage(
  message: StoredInputMessage,
): Promise<ZodDisplayInputMessage> {
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
): Promise<ZodDisplayInputMessageContent> {
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
        const fileContent: ZodFileContent = {
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

/**
 * Resolves a StoredInput to an Input with resolved file references.
 * Converts StoredFile (ObjectStoragePointer) to File with file_type: "object_storage".
 */
export async function resolveStoredInputToInput(
  storedInput: StoredInput,
): Promise<Input> {
  const resolvedMessages = await Promise.all(
    storedInput.messages.map(async (message) => {
      const resolvedContent = await Promise.all(
        message.content.map(async (content) => {
          return resolveStoredInputContentToInputContent(content);
        }),
      );
      return {
        role: message.role,
        content: resolvedContent,
      };
    }),
  );

  return {
    system: storedInput.system,
    messages: resolvedMessages,
  };
}

/**
 * Resolves a StoredInputMessageContent to InputMessageContent.
 * For files: converts StoredFile to File with file_type: "object_storage" by fetching data.
 */
async function resolveStoredInputContentToInputContent(
  content: StoredInputMessageContent,
): Promise<InputMessageContent> {
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
        // Fetch the file data from object storage
        const objectData = await getTensorZeroClient().getObject(
          content.storage_path,
        );
        const json = JSON.parse(objectData);
        const data = `data:${content.mime_type};base64,${json.data}`;

        // Return as File with file_type: "object_storage" and data
        return {
          type: "file",
          file_type: "object_storage",
          data: data,
          source_url: content.source_url,
          mime_type: content.mime_type,
          storage_path: content.storage_path,
          detail: content.detail,
          filename: content.filename,
        };
      } catch (error) {
        // On error, return as `object_storage_error`
        const errorMessage =
          error instanceof Error ? error.message : "Failed to fetch file data.";

        return {
          type: "file",
          file_type: "object_storage_error",
          source_url: content.source_url,
          mime_type: content.mime_type,
          storage_path: content.storage_path,
          detail: content.detail,
          filename: content.filename,
          error: errorMessage,
        };
      }
  }
}
