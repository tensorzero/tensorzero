import type { LoaderFunctionArgs } from "react-router";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { data } from "react-router";

export async function loader({ params }: LoaderFunctionArgs) {
  const { dataset_name, datapoint_id } = params;
  if (!dataset_name || !datapoint_id) {
    return data({ error: "dataset_name and id are required" }, 400);
  }

  try {
    const datapoint = await getDatapoint(dataset_name, datapoint_id);
    if (!datapoint) {
      return data({ error: `No datapoint found for id ${datapoint_id}` }, 404);
    }
    return data({ datapoint });
  } catch (error) {
    return data({ error: `Failed to fetch datapoint: ${error}` }, 500);
  }
}
