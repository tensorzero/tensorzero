import { useConfig } from "~/context/config";

// TODO replace with query/refetching
export const useMetric = (metricName: string) => {
  const config = useConfig();
  const metricConfig = config.metrics[metricName];
  return metricConfig;
};

// TODO
export const useFunction = (functionName: string) => {
  const config = useConfig();
  const functionConfig = config.functions[functionName];
  return functionConfig;
};
