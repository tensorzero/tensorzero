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
  namespace: string | undefined;
}): Promise<ThroughputSectionData> {
  const { function_name, time_granularity, namespace } = params;
  const tag = namespace ? `tensorzero::namespace::${namespace}` : undefined;

  const client = getTensorZeroClient();
  const response = await client.getFunctionThroughputByVariant(
    function_name,
    time_granularity,
    10,
    tag,
  );

  return response.throughput;
}
