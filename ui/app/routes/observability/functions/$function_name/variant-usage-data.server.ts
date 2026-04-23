import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow, VariantUsageTimePoint } from "~/types/tensorzero";

export async function fetchVariantUsageSectionData(params: {
  function_name: string;
  time_granularity: TimeWindow;
}): Promise<VariantUsageTimePoint[]> {
  const { function_name, time_granularity } = params;

  const client = getTensorZeroClient();
  const response = await client.getVariantUsageTimeseries(
    function_name,
    time_granularity,
    10,
  );

  return response.data;
}
