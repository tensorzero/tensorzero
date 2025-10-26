import {
  isTensorZeroServerError,
  TensorZeroServerError,
} from "~/utils/tensorzero";
import { isErrorLike, JSONParseError } from "~/utils/common";
import type { Route } from "./+types/inference";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";
import type { ClientInferenceParams } from "~/types/tensorzero";
import { getExtraInferenceOptions } from "~/utils/feature_flags";

export async function action({ request }: Route.ActionArgs): Promise<Response> {
  const formData = await request.formData();
  try {
    const data = formData.get("data");
    if (typeof data !== "string") {
      return Response.json({ error: "Missing request data" }, { status: 400 });
    }
    const parsed = JSON.parse(data);
    const extraOptions = getExtraInferenceOptions();
    const request = { ...parsed, ...extraOptions } as ClientInferenceParams;
    const nativeClient = await getNativeTensorZeroClient();
    const inference = await nativeClient.inference(request).catch((error) => {
      if (isErrorLike(error)) {
        throw new TensorZeroServerError(error.message, { status: 500 });
      }
      throw error;
    });
    return Response.json(inference);
  } catch (error) {
    if (process.env.NODE_ENV === "development") {
      console.error("Error in inference action:", error);
    }

    if (error instanceof JSONParseError) {
      return Response.json(
        { error: "Error parsing request data" },
        { status: 400 },
      );
    }
    if (isTensorZeroServerError(error)) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    return Response.json({ error: "Server error" }, { status: 500 });
  }
}
