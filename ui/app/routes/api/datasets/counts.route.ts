import { data, type LoaderFunctionArgs } from "react-router";
import { getDatasetMetadata } from "~/utils/clickhouse/datasets.server";

export async function loader({ request }: LoaderFunctionArgs) {
  try {
    const url = new URL(request.url);
    const functionName = url.searchParams.get("function") ?? undefined;
    const datasetMetadata = await getDatasetMetadata({
      function_name: functionName,
    });
    const datasets = datasetMetadata.map((d) => ({
      name: d.dataset_name,
      count: d.count,
      lastUpdated: d.last_updated,
    }));
    return data({ datasets });
  } catch (error) {
    return data({ error: `Failed to get count: ${error}` }, 500);
  }
}
