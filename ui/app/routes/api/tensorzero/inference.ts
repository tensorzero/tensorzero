import { ZodError } from "zod";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  isTensorZeroServerError,
  InferenceRequestSchema,
  type InferenceResponse,
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
