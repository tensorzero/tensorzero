import { ZodError } from "zod";
import {
  InferenceRequestSchema,
  HttpError as TensorZeroHttpError,
} from "~/utils/tensorzero";
import { tensorZeroClient } from "~/utils/tensorzero.server";
import type {
  ResolvedFileContent,
  ResolvedInput,
  ResolvedInputMessageContent,
} from "~/utils/clickhouse/common";
import type {
  InferenceResponse,
  Input as TensorZeroInput,
} from "~/utils/tensorzero";
import type { ResolvedInputMessage } from "~/utils/clickhouse/common";
import type { InputMessage as TensorZeroMessage } from "~/utils/tensorzero";
import type { InputMessageContent as TensorZeroContent } from "~/utils/tensorzero";
import type { ImageContent as TensorZeroImage } from "~/utils/tensorzero";
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
    if (error instanceof TensorZeroHttpError) {
      return Response.json(
        { error: error.message },
        { status: error.response.status },
      );
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

  return await tensorZeroClient.inference({
    ...result.data,
    stream: false,
  });
}

export function resolvedInputToTensorZeroInput(
  input: ResolvedInput,
): TensorZeroInput {
  return {
    ...input,
    messages: input.messages.map(resolvedInputMessageToTensorZeroMessage),
  };
}

function resolvedInputMessageToTensorZeroMessage(
  message: ResolvedInputMessage,
): TensorZeroMessage {
  return {
    ...message,
    content: message.content.map(
      resolvedInputMessageContentToTensorZeroContent,
    ),
  };
}

function resolvedInputMessageContentToTensorZeroContent(
  content: ResolvedInputMessageContent,
): TensorZeroContent {
  switch (content.type) {
    case "text":
      // If the text is string then send it as {"type": "text", "text": "..."}
      // If it is an object then send it as {"type": "text", "arguments": {...}}
      if (typeof content.value === "string") {
        return {
          type: "text",
          text: content.value,
        };
      }
      return {
        type: "text",
        arguments: content.value,
      };
    case "raw_text":
    case "tool_call":
    case "tool_result":
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
  const data = content.file.url.split(",")[1];
  return {
    type: "image",
    mime_type: content.file.mime_type,
    data,
  };
}
