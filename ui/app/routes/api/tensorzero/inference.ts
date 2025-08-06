import { isTensorZeroServerError } from "~/utils/tensorzero";
import { JSONParseError } from "~/utils/common";
import type { Route } from "./+types/inference";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";
import type { ClientInferenceParams } from "tensorzero-node";
import { getExtraInferenceOptions } from "~/utils/env.server";

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
    const inference = await nativeClient.inference(request);
    return Response.json(inference);
  } catch (error) {
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
