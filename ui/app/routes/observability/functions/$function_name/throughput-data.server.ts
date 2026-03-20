import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow } from "~/types/tensorzero";

export type ThroughputSectionData = Awaited<
  ReturnType<
    ReturnType<typeof getTensorZeroClient>["getFunctionThroughputByVariant"]
  >
>["throughput"];

export async function fetchThroughputSectionData(params: {
  function_name: string;
  time_granularity: TimeWindow;
}): Promise<ThroughputSectionData> {
  const { function_name, time_granularity } = params;

  const client = getTensorZeroClient();
  const response = await client.getFunctionThroughputByVariant(
    function_name,
    time_granularity,
    10,
  );

  return response.throughput;
}
