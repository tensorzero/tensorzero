import type {
  StoredInput,
  Input,
  InputMessageContent,
} from "~/types/tensorzero";

/**
 * Converts a StoredInput (with file storage pointers) to an Input (for API requests).
 * The backend will handle resolving files from storage.
 */
export function storedInputToTensorZeroInput(input: StoredInput): Input {
  return {
    ...input,
    messages: input.messages.map((message) => ({
      ...message,
      content: message.content.map((content): InputMessageContent => {
        if (content.type === "file") {
          // Convert StoredFile to File with file_type
          return {
            type: "file",
            file_type: "object_storage_pointer",
            source_url: content.source_url,
            mime_type: content.mime_type,
            storage_path: content.storage_path,
            detail: content.detail,
            filename: content.filename,
          };
        }

        // All other types are directly compatible
        return content;
      }),
    })),
  };
}
