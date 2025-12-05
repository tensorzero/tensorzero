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
  File,
  FunctionConfig,
  JsonValue,
  Input,
  InputContentBlock,
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

/**
 * Loads the content of files for an `Input`.
 * Converts `ObjectStoragePointer` to `File` with `file_type: "object_storage"`.
 *
 * TODO (#4674 #4675): This will be handled in the gateway.
 */
export async function loadFileDataForInput(input: Input): Promise<Input> {
  const resolvedMessages = await Promise.all(
    input.messages.map(async (message) => {
      const resolvedContent = await Promise.all(
        message.content.map(async (content) => {
          return loadFileDataForInputContent(content);
        }),
      );
      return {
        role: message.role,
        content: resolvedContent,
      };
    }),
  );

  return {
    system: input.system,
    messages: resolvedMessages,
  };
}

/**
 * Resolves a StoredInputContentBlock to InputContentBlock.
 * For files: converts StoredFile to File with file_type: "object_storage" by fetching data.
 */
async function loadFileDataForInputContent(
  content: InputContentBlock,
): Promise<InputContentBlock> {
  switch (content.type) {
    case "tool_call":
    case "tool_result":
    case "raw_text":
    case "thought":
    case "unknown":
    case "template":
    case "text":
      return content;
    case "file": {
      const loadedFile = await loadInputFileData(content);
      return {
        type: "file",
        ...loadedFile,
      };
    }
  }
}

/**
 * Loads the data of a `file`, converting `ObjectStoragePointer` to `ObjectStorage` or `ObjectStorageError`.
 * @param file - The file to load.
 * @returns Loaded file.
 */
async function loadInputFileData(file: File): Promise<File> {
  switch (file.file_type) {
    // These types should be input-only, and if we need to resolve then on the FE,
    // it represents an error (because we needed to store them).
    case "url":
    case "base64": {
      throw new Error(
        "URL and base64 files should not be passed to `loadInputFile`. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
      );
    }
    // These types are already resolved on the backend.
    case "object_storage":
      return file;
    case "object_storage_error":
      return file;
    // ObjectStoragePointer can be loaded in the UI.
    case "object_storage_pointer": {
      try {
        const fileContent: ZodFileContent = {
          type: "file",
          file: {
            url: file.source_url,
            mime_type: file.mime_type,
          },
          storage_path: file.storage_path,
        };
        const resolvedFile = await resolveFile(fileContent);
        const loadedFile: File = {
          file_type: "object_storage",
          data: resolvedFile.data,
          mime_type: resolvedFile.mime_type,
          storage_path: file.storage_path,
          source_url: file.source_url,
          detail: file.detail,
          filename: file.filename,
        };
        return loadedFile;
      } catch (error) {
        const loadFileError: File = {
          file_type: "object_storage_error",
          source_url: file.source_url,
          mime_type: file.mime_type,
          storage_path: file.storage_path,
          detail: file.detail,
          filename: file.filename,
          error: error instanceof Error ? error.message : String(error),
        };
        return loadFileError;
      }
    }
  }
}
