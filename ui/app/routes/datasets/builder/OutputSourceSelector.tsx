import { FormField, FormLabel } from "~/components/ui/form";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { useWatch, type Control } from "react-hook-form";
import type { DatasetBuilderFormValues } from "./types";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import { useEffect, useMemo } from "react";
import { Badge } from "~/components/ui/badge";
import clsx from "clsx";

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
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
          <FormLabel>Outputs</FormLabel>
          <div className="border border-border bg-bg-primary rounded-lg">
            <RadioGroup onValueChange={field.onChange} value={field.value} className="flex flex-col gap-0">
              <FormLabel
                htmlFor="none"
                className="flex items-center space-x-2 px-3 py-3 border-b border-border cursor-pointer"
              >
                <RadioGroupItem value="none" id="none" />
                <span>Without outputs</span>
              </FormLabel>
              <FormLabel
                htmlFor="inference"
                className="w-full flex items-center space-x-2 px-3 py-3 border-b border-border cursor-pointer"
              >
                <RadioGroupItem value="inference" id="inference" />
                <span className="w-full">With outputs from inference</span>
              </FormLabel>
              <FormLabel
                htmlFor="demonstration"
                className={clsx(
                  "flex items-center space-x-2 px-3 py-3",
                  demonstrationCount === 0
                    ? "opacity-50 cursor-not-allowed"
                    : "cursor-pointer",
                )}
              >
                <RadioGroupItem
                  value="demonstration"
                  id="demonstration"
                  disabled={demonstrationCount === 0}
                />
                <div className="flex items-center gap-2">
                  <span>With outputs from demonstration</span>
                  <Badge variant="secondary">
                    {demonstrationCount.toLocaleString()} available
                  </Badge>
                </div>
              </FormLabel>
            </RadioGroup>
          </div>
        </div>
      )}
    />
  );
}
