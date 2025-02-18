import { useForm, useWatch } from "react-hook-form";
import { Form } from "~/components/ui/form";
import { DatasetSelector } from "./DatasetSelector";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useConfig } from "~/context/config";
import { MetricSelector } from "~/components/metric/MetricSelector";
import { useCountFetcher } from "./route";

export function DatasetBuilderForm({
  dataset_counts,
}: {
  dataset_counts: DatasetCountInfo[];
}) {
  const config = useConfig();

  const form = useForm<DatasetBuilderFormValues>({
    defaultValues: {
      dataset_name: "",
      type: "chat",
      function_name: undefined,
      variant_name: undefined,
      metric_name: undefined,
      threshold: 0.5,
      join_demonstrations: false,
    },
    resolver: DatasetBuilderFormValuesResolver,
    mode: "onChange",
  });

  const watchedFields = useWatch({
    control: form.control,
    name: ["function_name", "metric_name", "threshold"] as const,
  });

  const [functionName, metricName, threshold] = watchedFields;
  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: threshold ?? undefined,
  });

  return (
    <Form {...form}>
      <form className="space-y-6">
        <div className="space-y-6">
          <DatasetSelector
            control={form.control}
            dataset_counts={dataset_counts}
          />
          <FunctionSelector<DatasetBuilderFormValues>
            control={form.control}
            name="function_name"
            inferenceCount={counts.inferenceCount}
            config={config}
          />
          <MetricSelector<DatasetBuilderFormValues>
            control={form.control}
            name="metric_name"
            functionFieldName="function_name"
            feedbackCount={counts.feedbackCount}
            curatedInferenceCount={counts.curatedInferenceCount}
            config={config}
          />
        </div>
      </form>
    </Form>
  );
}
