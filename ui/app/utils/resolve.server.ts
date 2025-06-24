import type {
  DisplayInput,
  DisplayInputMessage,
  DisplayInputMessageContent,
  DisplayMissingFunctionTextInput,
  DisplayUnstructuredTextInput,
  DisplayStructuredTextInput,
  FileContent,
  Input,
  InputMessage,
  InputMessageContent,
  ModelInferenceInputMessage,
  ModelInferenceInputMessageContent,
  ResolvedBase64File,
  Role,
  TextInput,
} from "./clickhouse/common";
import type { FunctionConfig } from "./config/function";
import { tensorZeroClient } from "./tensorzero.server";

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
          file: await resolveFile(content as FileContent),
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
        type: "unstructured_text",
        text: content.text,
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
          file: await resolveFile(content as FileContent),
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
  const object = await tensorZeroClient.getObject(content.storage_path);
  const json = JSON.parse(object);
  const dataURL = `data:${content.file.mime_type};base64,${json.data}`;
  return {
    dataUrl: dataURL,
    mime_type: content.file.mime_type,
  };
}

// In the current data model we can't distinguish between a message being a structured one from a schema
// or an unstructured one without a schema without knowing the function config.
// So as we prepare the input for display, we check this and return an unambiguous type of structured or unstructured text.
function prepareDisplayText(
  textBlock: TextInput,
  role: Role,
  functionConfig: FunctionConfig | null,
):
  | DisplayUnstructuredTextInput
  | DisplayStructuredTextInput
  | DisplayMissingFunctionTextInput {
  if (!functionConfig) {
    return {
      type: "missing_function_text",
      value: textBlock.value,
    };
  }

  // True if the function has a schema for the role (user or assistant)
  const hasSchemaForRole =
    role === "user"
      ? functionConfig.user_schema !== undefined
      : functionConfig.assistant_schema !== undefined;
  if (hasSchemaForRole) {
    return {
      type: "structured_text",
      arguments: textBlock.value,
    };
  }
  return {
    type: "unstructured_text",
    text: textBlock.value,
  };
}
