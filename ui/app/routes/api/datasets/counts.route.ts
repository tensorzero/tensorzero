import { data, type LoaderFunctionArgs } from "react-router";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";

export async function loader({ request }: LoaderFunctionArgs) {
  try {
    const url = new URL(request.url);
    console.log("datasets/counts");
    console.log(url.toString());
    const functionName = url.searchParams.get("function") ?? undefined;
    const datasetCounts = await getDatasetCounts({
      function_name: functionName,
    });
    const datasets = datasetCounts.map((d) => ({
      name: d.dataset_name,
      count: d.count,
      lastUpdated: d.last_updated,
    }));
    return data({ datasets });
  } catch (error) {
    return data({ error: `Failed to get count: ${error}` }, 500);
  }
}
