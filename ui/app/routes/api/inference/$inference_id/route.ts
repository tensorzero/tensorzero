import { data, type LoaderFunctionArgs } from "react-router";
import { queryInferenceById } from "~/utils/clickhouse/inference.server";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { inference_id } = params;
  
  if (!inference_id) {
    throw data("Inference ID is required", { status: 400 });
  }

  const inference = await queryInferenceById(inference_id);
  
  if (!inference) {
    throw data(`Inference ${inference_id} not found`, { status: 404 });
  }

  return Response.json(inference as ParsedInferenceRow);
}