import { FormField, FormLabel } from "~/components/ui/form";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { useWatch, type Control } from "react-hook-form";
import type { DatasetBuilderFormValues } from "./types";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import { useEffect, useMemo } from "react";
import { Badge } from "~/components/ui/badge";

export default function OutputSourceSelector({
  control,
}: {
  control: Control<DatasetBuilderFormValues>;
}) {
  const fieldName = "output_source";
  const functionFieldName = "function";
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();
  const functionValue = useWatch({
    control,
    name: functionFieldName,
  });
  useEffect(() => {
    if (functionValue && typeof functionValue === "string") {
      metricsFetcher.load(
        `/api/function/${encodeURIComponent(functionValue)}/feedback_counts`,
      );
    }
  }, [functionValue]);

  const demonstrationCount = useMemo(() => {
    if (!metricsFetcher.data) return 0;
    const demonstrationMetric = metricsFetcher.data.metrics.find(
      (m) => m.metric_type === "demonstration",
    );
    return demonstrationMetric?.feedback_count ?? 0;
  }, [metricsFetcher.data]);

  return (
    <FormField
      control={control}
      name={fieldName}
      render={({ field }) => (
        <div>
          <FormLabel>Outputs to be used in dataset</FormLabel>
          <div className="mt-2 grid gap-x-8 gap-y-2">
            <RadioGroup onValueChange={field.onChange} value={field.value}>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem value="none" id="none" />
                <FormLabel htmlFor="none">None</FormLabel>
              </div>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem value="inference" id="inference" />
                <FormLabel htmlFor="inference">Inference</FormLabel>
              </div>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem
                  value="demonstration"
                  id="demonstration"
                  disabled={demonstrationCount === 0}
                />
                <FormLabel
                  htmlFor="demonstration"
                  className={`flex items-center gap-2 ${
                    demonstrationCount === 0 ? "opacity-50" : ""
                  }`}
                >
                  Demonstration
                  <Badge variant="secondary">
                    {demonstrationCount.toLocaleString()} available
                  </Badge>
                </FormLabel>
              </div>
            </RadioGroup>
          </div>
        </div>
      )}
    />
  );
}
