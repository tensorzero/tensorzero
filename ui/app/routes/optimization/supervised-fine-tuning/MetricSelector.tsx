import { useWatch, useFormContext, type Control } from "react-hook-form";
import type { SFTFormValues } from "./types";
import { Config } from "~/utils/config";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Skeleton } from "~/components/ui/skeleton";
import { Input } from "~/components/ui/input";
import { MetricBadges } from "~/components/metric/MetricBadges";
import { useEffect, useMemo } from "react";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import { Badge } from "~/components/ui/badge";

type MetricSelectorProps = {
  control?: Control<SFTFormValues>;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  config: Config;
};

export function MetricSelector({
  control,
  feedbackCount,
  curatedInferenceCount,
  config,
}: MetricSelectorProps) {
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();
  const { getValues, setValue } = useFormContext<SFTFormValues>();

  const functionValue = useWatch({
    control,
    name: "function",
  });

  useEffect(() => {
    if (functionValue) {
      metricsFetcher.load(
        `/api/function/${encodeURIComponent(functionValue)}/feedback_counts`,
      );
    }
  }, [functionValue]);

  const validMetrics = useMemo(() => {
    if (!metricsFetcher.data) return new Set<string>();
    return new Set(metricsFetcher.data.metrics.map((m) => m.metric_name));
  }, [metricsFetcher.data]);

  const isLoading = metricsFetcher.state === "loading";

  // Reset metric value if the selected function does not have the previously selected metric
  useEffect(() => {
    const metricValue = getValues("metric");
    if (functionValue && metricValue && !validMetrics.has(metricValue)) {
      setValue("metric", null);
    }
  }, [functionValue, validMetrics, getValues, setValue]);

  return (
    <FormField
      control={control}
      name="metric"
      render={({ field }) => (
        <FormItem className="flex flex-col justify-center">
          <FormLabel>Metric</FormLabel>
          <div className="grid items-center gap-x-8 md:grid-cols-2">
            <div className="space-y-2">
              <Select
                onValueChange={(value: string) => {
                  const metricValue = value === "none" ? null : value;
                  field.onChange(metricValue);
                }}
                value={field.value ?? "none"}
                disabled={!functionValue || isLoading}
              >
                <SelectTrigger>
                  <SelectValue
                    placeholder={
                      isLoading ? "Loading metrics..." : "Select a metric"
                    }
                  />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">
                    <div className="flex w-full items-center justify-between">
                      <span>None</span>
                    </div>
                  </SelectItem>
                  {Object.entries(config.metrics)
                    .filter(([name]) => validMetrics.has(name))
                    .map(([name, metric]) => {
                      const metricFeedback = metricsFetcher.data?.metrics.find(
                        (m) => m.metric_name === name,
                      );

                      return (
                        <SelectItem key={name} value={name}>
                          <div className="flex w-full items-center justify-between">
                            <span>{name}</span>
                            <div className="ml-2 flex items-center gap-2">
                              {metricFeedback && (
                                <Badge className="bg-gray-200 text-gray-800 dark:bg-gray-900 dark:text-gray-300">
                                  Count: {metricFeedback.feedback_count}
                                </Badge>
                              )}
                              <MetricBadges metric={metric} />
                            </div>
                          </div>
                        </SelectItem>
                      );
                    })}
                </SelectContent>
              </Select>

              {field.value && config.metrics[field.value]?.type === "float" && (
                <FormField
                  control={control}
                  name="threshold"
                  render={({ field: thresholdField }) => (
                    <div className="rounded-lg bg-gray-100 p-4">
                      <FormLabel>Threshold</FormLabel>
                      <Input
                        type="number"
                        step="0.01"
                        {...thresholdField}
                        className="border-none bg-transparent focus:ring-0"
                        onChange={(e) => {
                          thresholdField.onChange(Number(e.target.value));
                        }}
                      />
                    </div>
                  )}
                />
              )}
            </div>

            <div className="space-y-1 text-sm text-muted-foreground">
              <div>
                Feedbacks:{" "}
                {/* If field.value is empty string (unselected), show loading skeleton */}
                {field.value === "" ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is null (selected "None"), show N/A */
                field.value === null ? (
                  <span className="font-medium">N/A</span>
                ) : (
                  /* Otherwise show the actual feedback count */
                  <span className="font-medium">{feedbackCount}</span>
                )}
              </div>
              <div>
                Curated Inferences:{" "}
                {/* If field.value is empty string (unselected), show loading skeleton */}
                {field.value === "" ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is null (selected "None"), show N/A */
                field.value === null ? (
                  <span className="font-medium">N/A</span>
                ) : (
                  /* Otherwise show the actual curated inference count */
                  <span className="font-medium">{curatedInferenceCount}</span>
                )}
              </div>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
