import {
  useFormContext,
  useWatch,
  type Control,
  type Path,
  type PathValue,
} from "react-hook-form";
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
import MetricBadges from "~/components/metric/MetricBadges";
import { useEffect, useMemo } from "react";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import { Badge } from "~/components/ui/badge";

type CurationMetricSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  functionFieldName: Path<T>;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  removeDemonstrations?: boolean;
  config: Config;
};

/**
 * This component is used to select a metric for a function.
 * It is used in the DatasetBuilderForm and SFTForm, where we are curating a dataset for either
 * fine-tuning or just to build the dataset.
 * You should use this component if you're using react-hook-form and you want to display the counts
 * of feedbacks and curated inferences for the selected metric.
 *
 * In the future we should refactor this so it works in this context as well as in the
 * context used in the feedback modal and the selection for variants.
 */
export default function CurationMetricSelector<
  T extends Record<string, unknown>,
>({
  control,
  name,
  functionFieldName,
  feedbackCount,
  curatedInferenceCount,
  config,
  removeDemonstrations = false,
}: CurationMetricSelectorProps<T>) {
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();
  const { getValues, setValue } = useFormContext<T>();

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

  const validMetrics = useMemo(() => {
    if (!metricsFetcher.data) return new Set<string>();
    return new Set(
      metricsFetcher.data.metrics
        .filter(
          (m) => !removeDemonstrations || m.metric_name !== "demonstration",
        )
        .map((m) => m.metric_name),
    );
  }, [metricsFetcher.data, removeDemonstrations]);

  const isLoading = metricsFetcher.state === "loading";

  // Reset metric value if the selected function does not have the previously selected metric
  useEffect(() => {
    const metricValue = getValues(name);
    if (
      functionValue &&
      metricValue &&
      typeof metricValue === "string" &&
      !validMetrics.has(metricValue)
    ) {
      // TODO: Figure out how to generalize the generic for this function so that it accepts a null value
      setValue(name, null as PathValue<T, Path<T>>);
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [functionValue, validMetrics, getValues, setValue]);

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex w-full flex-col gap-1">
          <FormLabel>Metric</FormLabel>
          <div className="space-y-2">
            <Select
              onValueChange={(value: string) => {
                const metricValue = value === "none" ? null : value;
                field.onChange(metricValue);
              }}
              value={(field.value ?? "none") as string}
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
                {Object.entries(config.metrics).map(([name, metric]) => {
                  if (removeDemonstrations && name === "demonstration") {
                    return null;
                  }

                  const metricFeedback = metricsFetcher.data?.metrics.find(
                    (m) => m.metric_name === name,
                  );

                  return (
                    <SelectItem key={name} value={name}>
                      <div className="flex w-full items-center justify-between">
                        <span>{name}</span>
                        <div className="ml-2 flex items-center gap-2">
                          <Badge className="bg-gray-200 text-gray-800 dark:bg-gray-900 dark:text-gray-300">
                            Count:{" "}
                            {metricFeedback ? metricFeedback.feedback_count : 0}
                          </Badge>
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
                name={"threshold" as Path<T>}
                render={({ field: thresholdField }) => (
                  <div className="rounded-lg bg-gray-100 p-4">
                    <FormLabel>Threshold</FormLabel>
                    <Input
                      type="number"
                      step="0.01"
                      {...thresholdField}
                      value={thresholdField.value?.toString() ?? ""}
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

          <div className="text-muted-foreground space-y-1 text-sm">
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
        </FormItem>
      )}
    />
  );
}
