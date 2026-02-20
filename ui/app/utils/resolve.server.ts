/**
 * Server-side utilities for resolving file content for inputs and model inferences.
 *
 * TODO(shuyangli): find a nice way to avoid duplicating all the types with resolved files; possibly lazily load them.
 */

import type {
  ZodFileContent,
  ZodResolvedBase64File,
} from "./clickhouse/common";
import type {
  File,
  InputMessage,
  Input,
  InputMessageContent,
  StoredInput,
  StoredInputMessageContent,
  StoredFile,
  ModelInference,
  StoredRequestMessage,
  StoredContentBlock,
} from "~/types/tensorzero";
import { getTensorZeroClient } from "./tensorzero.server";
import type { ParsedModelInferenceRow } from "./clickhouse/inference";

/**
 * Resolves model inferences by transforming input_messages for display.
 * This fetches file content from object storage and prepares messages for rendering.
 */
export async function resolveModelInferences(
  modelInferences: ModelInference[],
): Promise<ParsedModelInferenceRow[]> {
  return Promise.all(
    modelInferences.map(async (row) => {
      const resolvedMessages = await resolveModelInferenceMessages(
        row.input_messages,
      );
      return {
        ...row,
        input_messages: resolvedMessages,
        output: row.output ?? [],
      } as ParsedModelInferenceRow;
    }),
  );
}

async function resolveModelInferenceMessages(
  messages: StoredRequestMessage[] | undefined,
): Promise<InputMessage[]> {
  if (!messages) {
    return [];
  }
  return Promise.all(
    messages.map(async (message) => {
      return resolveModelInferenceMessage(message);
    }),
  );
}

async function resolveModelInferenceMessage(
  message: StoredRequestMessage,
): Promise<InputMessage> {
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

async function resolveModelInferenceContent(
  content: StoredContentBlock,
): Promise<InputMessageContent> {
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
    case "thought":
    case "unknown":
      return content;
    case "file": {
      const fileContent: ZodFileContent = {
        type: "file",
        file: {
          mime_type: content.mime_type,
          url: content.source_url,
        },
        storage_path: content.storage_path,
      };
      try {
        return {
          ...content,
          file_type: "object_storage",
          data: (await resolveFile(fileContent)).data,
        };
      } catch (error) {
        return {
          ...content,
          file_type: "object_storage_error",
          error: error instanceof Error ? error.message : String(error),
        };
      }
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
    system: input.system ?? undefined,
    messages: resolvedMessages,
  };
}

/**
 * Resolves a StoredInputMessageContent to InputMessageContent.
 * For files: converts StoredFile to File with file_type: "object_storage" by fetching data.
 */
async function loadFileDataForInputContent(
  content: InputMessageContent,
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

/**
 * Loads the content of files for an `StoredInput`.
 * Converts `ObjectStoragePointer` to `File` with `file_type: "object_storage"`.
 *
 * TODO (#4674 #4675): This will be handled in the gateway.
 */
export async function loadFileDataForStoredInput(
  input: StoredInput,
): Promise<Input> {
  const resolvedMessages = await Promise.all(
    input.messages.map(async (message) => {
      const resolvedContent = await Promise.all(
        message.content.map(async (content) => {
          return loadFileDataForStoredInputContent(content);
        }),
      );
      return {
        role: message.role,
        content: resolvedContent,
      };
    }),
  );

  return {
    system: input.system ?? undefined,
    messages: resolvedMessages,
  };
}

/**
 * Resolves a StoredInputMessageContent to InputMessageContent.
 * For files: converts StoredFile to File with file_type: "object_storage" by fetching data.
 */
async function loadFileDataForStoredInputContent(
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
    case "file": {
      const loadedFile = await loadStoredInputFileData(content);
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
async function loadStoredInputFileData(file: StoredFile): Promise<File> {
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
