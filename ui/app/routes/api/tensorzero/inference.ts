import type { ActionFunctionArgs } from "react-router";
import { InferenceRequestSchema } from "~/utils/tensorzero";
import { tensorZeroClient } from "~/utils/tensorzero.server";

export async function action({
  request,
}: ActionFunctionArgs): Promise<Response> {
  const formData = await request.formData();
  const rawData = JSON.parse(formData.get("data") as string);
  console.log("rawData", rawData);
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
