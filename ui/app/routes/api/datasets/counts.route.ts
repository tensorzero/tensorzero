import { data } from "react-router";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";

export async function loader() {
  try {
    const datasetCounts = await getDatasetCounts();
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
