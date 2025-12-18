import { data, type LoaderFunctionArgs } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: LoaderFunctionArgs) {
  try {
    const url = new URL(request.url);
    const functionName = url.searchParams.get("function") ?? undefined;
    const datasetMetadata = await getTensorZeroClient().listDatasets({
      function_name: functionName,
    });
    const datasets = datasetMetadata.datasets.map((d) => ({
      name: d.dataset_name,
      count: d.datapoint_count,
      lastUpdated: d.last_updated,
    }));
    return data({ datasets });
  } catch (error) {
    return data({ error: `Failed to get count: ${error}` }, 500);
  }
}
