import { useQuery } from "@tanstack/react-query";
import { z } from "zod";

export const DatasetCountResponse = z.object({
  datasets: z.array(
    z.object({
      name: z.string(),
      count: z.number(),
      lastUpdated: z.string().datetime(),
    }),
  ),
});

export const useDatasetCounts = (functionName?: string) => {
  return useQuery({
    queryKey: ["DATASETS_COUNT", functionName],
    queryFn: async ({ signal }) => {
      const url = new URL("/api/datasets/counts", window.location.origin);
      if (functionName) {
        url.searchParams.append("function", functionName);
      }
      const response = await fetch(url.toString(), { signal });
      const data = await response.json();
      const parsedData = DatasetCountResponse.parse(data);
      return parsedData.datasets;
    },
  });
};
