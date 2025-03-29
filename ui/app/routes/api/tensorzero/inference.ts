import type { ActionFunctionArgs } from "react-router";
import { InferenceRequestSchema } from "~/utils/tensorzero";
import { tensorZeroClient } from "~/utils/tensorzero.server";
import type {
  ResolvedImageContent,
  ResolvedInput,
  ResolvedInputMessageContent,
} from "~/utils/clickhouse/common";
import type { Input as TensorZeroInput } from "~/utils/tensorzero";
import type { ResolvedInputMessage } from "~/utils/clickhouse/common";
import type { InputMessage as TensorZeroMessage } from "~/utils/tensorzero";
import type { InputMessageContent as TensorZeroContent } from "~/utils/tensorzero";
import type { ImageContent as TensorZeroImage } from "~/utils/tensorzero";

export async function action({
  request,
}: ActionFunctionArgs): Promise<Response> {
  const formData = await request.formData();
  const rawData = JSON.parse(formData.get("data") as string);
  const result = InferenceRequestSchema.safeParse(rawData);
  if (!result.success) {
    return Response.json({ error: result.error.issues }, { status: 400 });
  }

  const inference = await tensorZeroClient.inference({
    ...result.data,
    stream: false,
  });
  return Response.json(inference);
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
    case "raw_text":
    case "tool_call":
    case "tool_result":
      return content;
    case "image":
      return resolvedImageContentToTensorZeroImage(content);
    case "image_error":
      throw new Error("Can't convert image error to tensorzero content");
  }
}

function resolvedImageContentToTensorZeroImage(
  content: ResolvedImageContent,
): TensorZeroImage {
  const data = content.image.url.split(",")[1];
  return {
    type: "image",
    mime_type: content.image.mime_type,
    data,
  };
}
