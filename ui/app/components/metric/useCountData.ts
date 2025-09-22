import { useCountFetcher } from "~/routes/api/curated_inferences/count.route";

export function useCountData({
  functionName,
  metricName,
  parsedThreshold,
}: {
  functionName: string | null;
  metricName: string | null;
  parsedThreshold: number;
}) {
  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: !isNaN(parsedThreshold) ? parsedThreshold : undefined,
  });

  const isCuratedInferenceCountLow =
    counts.curatedInferenceCount !== null && counts.curatedInferenceCount < 10;

  return { counts, isCuratedInferenceCountLow };
}
