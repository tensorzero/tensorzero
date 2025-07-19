import { ZodError } from "zod";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type {
  ResolvedFileContent,
  DisplayInput,
  DisplayInputMessageContent,
} from "~/utils/clickhouse/common";
import type {
  InferenceResponse,
  InputMessageContent as TensorZeroContent,
  ImageContent as TensorZeroImage,
  InputMessage as TensorZeroMessage,
  Input as TensorZeroInput,
} from "~/utils/tensorzero";
import type { DisplayInputMessage } from "~/utils/clickhouse/common";
import {
  isTensorZeroServerError,
  InferenceRequestSchema,
} from "~/utils/tensorzero";
import { JSONParseError } from "~/utils/common";
import type { Route } from "./+types/inference";

export async function action({ request }: Route.ActionArgs): Promise<Response> {
  const formData = await request.formData();
  try {
    const inference = await handleInferenceAction(formData.get("data"));
    return Response.json(inference);
  } catch (error) {
    if (error instanceof JSONParseError) {
      return Response.json(
        { error: "Error parsing request data" },
        { status: 400 },
      );
    }
    if (error instanceof ZodError) {
      return Response.json({ error: error.issues }, { status: 400 });
    }
    if (isTensorZeroServerError(error)) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    return Response.json({ error: "Server error" }, { status: 500 });
  }
}

async function handleInferenceAction(
  payload: unknown,
): Promise<InferenceResponse> {
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch {
      throw new JSONParseError("Error parsing request data");
    }
  }

  const result = InferenceRequestSchema.safeParse(payload);
  if (!result.success) {
    throw result.error;
  }

  return await getTensorZeroClient().inference({
    ...result.data,
    stream: false,
  });
}

export function resolvedInputToTensorZeroInput(
  input: DisplayInput,
): TensorZeroInput {
  return {
    ...input,
    messages: input.messages.map(resolvedInputMessageToTensorZeroMessage),
  };
}

function resolvedInputMessageToTensorZeroMessage(
  message: DisplayInputMessage,
): TensorZeroMessage {
  return {
    ...message,
    content: message.content.map(
      resolvedInputMessageContentToTensorZeroContent,
    ),
  };
}

function resolvedInputMessageContentToTensorZeroContent(
  content: DisplayInputMessageContent,
): TensorZeroContent {
  switch (content.type) {
    case "structured_text":
      return {
        type: "text",
        arguments: content.arguments,
      };
    case "unstructured_text":
      return {
        type: "text",
        text: content.text,
      };
    case "missing_function_text":
      return {
        type: "text",
        text: content.value,
      };
    case "raw_text":
    case "tool_call":
    case "tool_result":
    case "thought":
    case "unknown":
      return content;
    case "file":
      return resolvedFileContentToTensorZeroFile(content);
    case "file_error":
      throw new Error("Can't convert image error to tensorzero content");
  }
}

function resolvedFileContentToTensorZeroFile(
  content: ResolvedFileContent,
): TensorZeroImage {
  const data = content.file.dataUrl.split(",")[1];
  return {
    type: "image",
    mime_type: content.file.mime_type,
    data,
  };
}
