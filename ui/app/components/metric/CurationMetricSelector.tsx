import {
  useFormContext,
  useWatch,
  type Control,
  type Path,
  type PathValue,
} from "react-hook-form";
import type { UiConfig } from "~/types/tensorzero";
import {
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
} from "~/components/ui/form";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { ChevronDown, Loader2 } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import { Input } from "~/components/ui/input";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import { useEffect, useId, useMemo, useState } from "react";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import clsx from "clsx";
import type { FeedbackConfig } from "~/utils/config/feedback";
import { Skeleton } from "../ui/skeleton";

type CurationMetricSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  functionFieldName: Path<T>;
  config: UiConfig;
  addDemonstrations: boolean;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  isLoading?: boolean;
  // Notifies parent when this component's internal fetcher is loading
  onMetricsLoadingChange?: (loading: boolean) => void;
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
  config,
  addDemonstrations,
  feedbackCount,
  curatedInferenceCount,
  isLoading = false,
  onMetricsLoadingChange,
}: CurationMetricSelectorProps<T>) {
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();
  const { getValues, setValue } = useFormContext<T>();
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const metrics = Object.fromEntries(
    Object.entries(config.metrics).filter(([, v]) => v !== undefined),
  ) as Record<string, FeedbackConfig>;

  if (addDemonstrations) {
    metrics["demonstration"] = {
      type: "demonstration",
      level: "inference",
    };
  }

  const functionValue = useWatch({
    control,
    name: functionFieldName,
  });

  const handleInputChange = (input: string) => {
    setInputValue(input);
  };

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
        .filter((m) => addDemonstrations || m.metric_name !== "demonstration")
        .map((m) => m.metric_name),
    );
  }, [metricsFetcher.data, addDemonstrations]);

  const metricsLoading = metricsFetcher.state === "loading";

  const metricLabelId = useId();
  const metricComboboxId = `${metricLabelId}-combobox`;

  // Inform parent when the internal metrics fetcher loading state changes
  useEffect(() => {
    onMetricsLoadingChange?.(metricsLoading);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metricsLoading]);

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
        <FormItem className="flex flex-col justify-center">
          <FormLabel id={`${metricLabelId}-label`} htmlFor={metricComboboxId}>
            Metric
          </FormLabel>
          <div className="items-top grid gap-x-8 md:grid-cols-2">
            <div className="space-y-2">
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    aria-busy={metricsLoading}
                    aria-label="Metric"
                    aria-labelledby={`${metricLabelId}-label`}
                    id={metricComboboxId}
                    className="group border-border hover:border-border-accent hover:bg-bg-primary w-full justify-between border px-3 font-normal hover:cursor-pointer"
                    disabled={!functionValue || metricsLoading}
                  >
                    <div className="flex min-w-0 flex-1 items-center gap-2">
                      {metricsLoading && (
                        <Loader2
                          className="text-fg-tertiary h-4 w-4 shrink-0 animate-spin"
                          aria-hidden
                        />
                      )}
                      {(() => {
                        const currentMetricName = field.value as string | null;
                        const selectedMetricDetails = currentMetricName
                          ? metrics[currentMetricName]
                          : undefined;

                        if (currentMetricName && selectedMetricDetails) {
                          return (
                            <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                              <span className="truncate text-sm">
                                {currentMetricName}
                              </span>
                            </div>
                          );
                        } else if (currentMetricName === null) {
                          return (
                            <div className="text-fg-muted flex items-center gap-x-2">
                              <span className="text-fg-secondary flex text-sm">
                                None
                              </span>
                            </div>
                          );
                        } else {
                          return (
                            <div className="text-fg-muted flex items-center gap-x-2">
                              <span className="text-fg-secondary flex text-sm">
                                {metricsLoading
                                  ? "Loading metrics..."
                                  : !functionValue ||
                                      (functionValue && validMetrics.size === 0)
                                    ? "No metrics available"
                                    : "Select a metric"}
                              </span>
                            </div>
                          );
                        }
                      })()}
                    </div>
                    <ChevronDown
                      className={clsx(
                        "text-fg-muted group-hover:text-fg-tertiary ml-2 h-4 w-4 shrink-0 transition-transform duration-300 ease-out",
                        open ? "-rotate-180" : "rotate-0",
                      )}
                    />
                  </Button>
                </PopoverTrigger>
                <PopoverContent
                  className="w-[var(--radix-popover-trigger-width)] p-0"
                  align="start"
                >
                  <Command>
                    <CommandInput
                      placeholder="Find a metric..."
                      value={inputValue}
                      onValueChange={handleInputChange}
                      className="h-9"
                    />
                    <CommandList>
                      <CommandEmpty className="px-4 py-2 text-sm">
                        No metrics found.
                      </CommandEmpty>
                      <CommandGroup
                        heading={
                          <div className="text-fg-tertiary flex w-full items-center justify-between">
                            <span>Feedback</span>
                            <span>Count</span>
                          </div>
                        }
                      >
                        <CommandItem
                          key="none"
                          value="none"
                          onSelect={() => {
                            const metricValue = null;
                            field.onChange(metricValue);
                            setInputValue("");
                            setOpen(false);
                          }}
                          className="group flex w-full cursor-pointer items-center justify-between"
                        >
                          <span>None</span>
                          <span
                            className={clsx(
                              "min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap",
                              field.value === null
                                ? "text-fg-secondary font-medium"
                                : "text-fg-tertiary font-normal",
                            )}
                          >
                            N/A
                          </span>
                        </CommandItem>
                        {Object.entries(metrics)
                          .sort(([metricNameA], [metricNameB]) => {
                            // 1. Put selectable metrics first
                            const isSelectableA = validMetrics.has(metricNameA);
                            const isSelectableB = validMetrics.has(metricNameB);
                            if (isSelectableA && !isSelectableB) {
                              return -1;
                            }
                            if (!isSelectableA && isSelectableB) {
                              return 1;
                            }
                            // 2. Within each category, put demonstration first if present
                            if (metricNameA === "demonstration") return -1;
                            if (metricNameB === "demonstration") return 1;
                            return 0;
                          })
                          .map(([metricName, metricConfig]) => {
                            const isSelectable = validMetrics.has(metricName);
                            const metricFeedback =
                              metricsFetcher.data?.metrics.find(
                                (m) => m.metric_name === metricName,
                              );
                            return (
                              <CommandItem
                                key={metricName}
                                value={metricName}
                                disabled={!isSelectable}
                                onSelect={() => {
                                  if (!isSelectable) return;
                                  field.onChange(metricName);
                                  setInputValue("");
                                  setOpen(false);
                                }}
                                className={clsx(
                                  "group flex w-full items-center justify-between",
                                  isSelectable
                                    ? "cursor-pointer"
                                    : "cursor-not-allowed opacity-50",
                                )}
                              >
                                <span className="truncate">{metricName}</span>
                                <div className="ml-2 flex items-center gap-2">
                                  <FeedbackBadges metric={metricConfig} />
                                  <span
                                    className={clsx(
                                      "min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap",
                                      field.value === metricName
                                        ? "text-fg-secondary font-medium"
                                        : "text-fg-tertiary font-normal",
                                    )}
                                  >
                                    {metricFeedback
                                      ? metricFeedback.feedback_count.toLocaleString()
                                      : "0"}
                                  </span>
                                </div>
                              </CommandItem>
                            );
                          })}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>

              {field.value && metrics[field.value]?.type === "float" && (
                <FormField
                  control={control}
                  name={"threshold" as Path<T>}
                  render={({ field: thresholdField }) => (
                    <FormItem className="flex flex-col gap-1 border-l pl-4">
                      <FormLabel>Threshold</FormLabel>
                      <FormControl>
                        <Input
                          type="text"
                          placeholder="0.5"
                          {...thresholdField}
                          value={thresholdField.value?.toString() ?? ""}
                          onChange={(e) => {
                            const val = e.target.value;

                            // Allow empty or partial inputs like "-" or "."
                            // Allow empty or partial inputs like "-", ".", "0.", "-1", etc.
                            if (val === "" || /^-?\d*\.?\d*$/.test(val)) {
                              thresholdField.onChange(val);
                              return;
                            }

                            const parsed = parseFloat(val);
                            if (!isNaN(parsed)) {
                              thresholdField.onChange(val);
                            } else {
                              // Do not store completely invalid values
                            }
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}
            </div>

            <div className="text-muted-foreground space-y-1 text-sm">
              <div>
                Feedbacks: {/* If data is loading, show skeleton */}
                {isLoading ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is empty string (unselected), show loading skeleton */
                field.value === "" ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is null (selected "None"), show N/A */
                field.value === null ? (
                  <span className="font-medium">N/A</span>
                ) : (
                  /* Otherwise show the actual feedback count */
                  <span className="font-medium">
                    {feedbackCount?.toLocaleString() ?? "0"}
                  </span>
                )}
              </div>
              <div>
                Curated Inferences: {/* If data is loading, show skeleton */}
                {isLoading ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is empty string (unselected), show loading skeleton */
                field.value === "" ? (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                ) : /* If field.value is null (selected "None"), show N/A */
                field.value === null ? (
                  <span className="font-medium">N/A</span>
                ) : (
                  /* Otherwise show the actual curated inference count */
                  <span className="font-medium">
                    {curatedInferenceCount?.toLocaleString() ?? "0"}
                  </span>
                )}
              </div>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
