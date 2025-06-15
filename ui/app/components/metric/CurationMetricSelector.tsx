import {
  useFormContext,
  useWatch,
  type Control,
  type Path,
  type PathValue,
} from "react-hook-form";
import { Config } from "~/utils/config";
import {
  FormField,
  FormItem,
  FormLabel,
} from "~/components/ui/form";
import { Input } from "~/components/ui/input";
import MetricBadges from "~/components/metric/MetricBadges";
import { useEffect, useMemo, useRef, useState } from "react";
import { useFetcher } from "react-router";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { useClickOutside } from "~/hooks/use-click-outside";

type CurationMetricSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  functionFieldName: Path<T>;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  totalFunctionInferenceCount?: number | null;
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
  totalFunctionInferenceCount,
  config,
}: CurationMetricSelectorProps<T>) {
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();
  const { getValues, setValue } = useFormContext<T>();
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const commandRef = useRef<HTMLDivElement>(null);

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
          (m) =>
            m.metric_name !== "demonstration" && m.metric_name !== "comment",
        )
        .map((m) => m.metric_name),
    );
  }, [metricsFetcher.data]);

  useClickOutside(commandRef, () => setOpen(false));

  const handleInputChange = (input: string) => {
    setInputValue(input);
    if (input.trim() !== "" && !open) {
      setOpen(true);
    }
  };

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

  // Determine if the selector should be disabled and what cursor to show
  const isSelectorDisabled = 
    !functionValue || 
    isLoading || 
    (functionValue && !isLoading && validMetrics.size === 0);

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="w-full flex flex-col gap-1">
            <FormLabel>Metric</FormLabel>
            <div className="relative h-10">
              <div
                ref={commandRef}
                className={clsx(
                  "absolute top-0 left-0 right-0 z-48 rounded-lg border border-border bg-background transition-shadow transition-transform ease-out duration-300",
                  open
                    ? "shadow-2xl"
                    : "hover:shadow-xs active:shadow-none active:scale-99 scale-100 shadow-none",
                )}
              >
                <Button
                  variant="ghost"
                  role="combobox"
                  type="button"
                  aria-expanded={open}
                  className={clsx(
                    "w-full px-3 hover:bg-transparent font-normal group cursor-pointer",
                    isSelectorDisabled && "cursor-not-allowed"
                  )}
                  onClick={() => setOpen(!open)}
                  disabled={isSelectorDisabled}
                >
                  <div className="min-w-0 flex-1">
                    {(() => {
                      const currentMetricName = field.value as string | null;
                      const selectedMetricDetails = currentMetricName ? config.metrics[currentMetricName] : undefined;

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
                          <div className="flex items-center gap-x-2 text-fg-muted">
                            <span className="text-fg-secondary flex text-sm">
                              None
                            </span>
                          </div>
                        );
                      } else {
                        return (
                          <div className="flex items-center gap-x-2 text-fg-muted">
                            <span className="text-fg-secondary flex text-sm">
                              {isLoading
                                ? "Loading metrics..."
                                : !functionValue || (functionValue && validMetrics.size === 0)
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
                      "ml-2 h-4 w-4 shrink-0 text-fg-muted group-hover:text-fg-tertiary transition-colors transition-transform ease-out duration-300",
                      open ? "-rotate-180" : "rotate-0",
                    )}
                  />
                </Button>
                <Command
                  className={clsx(
                    "border-t border-border rounded-none bg-transparent overflow-hidden transition-all ease-out duration-300",
                    open ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0",
                  )}
                >
                  <CommandInput
                    placeholder="Find a metric..."
                    value={inputValue}
                    onValueChange={handleInputChange}
                  />
                  <CommandList>
                    <CommandEmpty className="px-4 py-2 text-sm">
                      No metrics found.
                    </CommandEmpty>
                    <CommandGroup
                      heading={
                        <div className="text-fg-tertiary flex w-full items-center justify-between">
                          <span>Function Metrics</span>
                          <span>Inferences</span>
                        </div>
                      }
                    >
                      <CommandItem
                        key="none"
                        value="none"
                        onSelect={() => {
                          field.onChange(null);
                          setInputValue("");
                          setOpen(false);
                        }}
                        className="group flex w-full items-center justify-between cursor-pointer"
                      >
                        <span>None</span>
                        {totalFunctionInferenceCount !== null && typeof totalFunctionInferenceCount !== 'undefined' && (
                          <span
                            className={clsx(
                              "min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap text-fg-tertiary font-normal",
                            )}
                          >
                            {totalFunctionInferenceCount.toLocaleString()}
                          </span>
                        )}
                      </CommandItem>
                      {Object.entries(config.metrics)
                        .filter(
                          ([metricName]) =>
                            metricName !== "demonstration" &&
                            metricName !== "comment",
                        )
                        .sort(([metricNameA], [metricNameB]) => {
                          const isSelectableA = validMetrics.has(metricNameA);
                          const isSelectableB = validMetrics.has(metricNameB);
                          if (isSelectableA && !isSelectableB) {
                            return -1;
                          }
                          if (!isSelectableA && isSelectableB) {
                            return 1;
                          }
                          return 0;
                        })
                        .map(([metricName, metricConfig]) => {
                          const isSelectable = validMetrics.has(metricName);
                          const metricFeedback = metricsFetcher.data?.metrics.find(
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
                                isSelectable ? "cursor-pointer" : "opacity-50 cursor-not-allowed",
                              )}
                            >
                              <span className="truncate">{metricName}</span>
                              <div className="ml-2 flex items-center gap-2">
                                <MetricBadges metric={metricConfig} />
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
              </div>
            </div>

            {field.value && config.metrics[field.value as string]?.type === "float" && (
              <FormField
                control={control}
                name={"threshold" as Path<T>}
                render={({ field: thresholdField }) => (
                  <FormItem className="flex flex-col gap-1 pt-2">
                    <FormLabel>Threshold</FormLabel>
                    <Input
                      type="number"
                      step="0.01"
                      {...thresholdField}
                      value={thresholdField.value?.toString() ?? ""}
                      onChange={(e) => {
                        thresholdField.onChange(Number(e.target.value));
                        }}
                    />
                  </FormItem>
                )}
              />
            )}
        </FormItem>
      )}
    />
  );
}
