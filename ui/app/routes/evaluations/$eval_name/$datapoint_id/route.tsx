import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const eval_name = params.eval_name;
  const datapoint_id = params.datapoint_id;
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const { eval_name, datapoint_id } = loaderData;
}
