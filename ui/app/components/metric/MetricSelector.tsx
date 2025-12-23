import type { FeedbackConfig } from "~/utils/config/feedback";
import { useId, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";

import { Button } from "../ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import clsx from "clsx";

interface MetricSelectorProps {
  // The metrics are a record of metric name to metric config.
  metrics: { [x: string]: FeedbackConfig | undefined };
  selectedMetric: string | undefined; // Allow undefined for placeholder
  onMetricChange: (metric: string) => void;
  showLevelBadges?: boolean;
}

export default function MetricSelector({
  metrics,
  selectedMetric,
  onMetricChange,
  showLevelBadges = true,
}: MetricSelectorProps) {
  const [metricPopoverOpen, setMetricPopoverOpen] = useState(false);
  const [metricInputValue, setMetricInputValue] = useState("");

  const metricEntries = Object.entries(metrics);
  const hasMetrics = metricEntries.length > 0;
  const metricLabelId = useId();
  const metricComboboxId = `${metricLabelId}-combobox`;

  const filteredMetrics = metricInputValue
    ? metricEntries.filter(([name]) =>
        name.toLowerCase().includes(metricInputValue.toLowerCase()),
      )
    : metricEntries;

  return (
    <div>
      <div className="mt-4">
        <label
          id={`${metricLabelId}-label`}
          htmlFor={metricComboboxId}
          className="mb-1 block text-sm font-medium"
        >
          Metric
        </label>
      </div>
      <Popover open={metricPopoverOpen} onOpenChange={setMetricPopoverOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={metricPopoverOpen}
            aria-label="Metric"
            aria-labelledby={`${metricLabelId}-label`}
            id={metricComboboxId}
            className="w-full justify-between font-normal"
            disabled={!hasMetrics}
          >
            {selectedMetric
              ? selectedMetric // Display selected metric name
              : hasMetrics
                ? "Select a metric"
                : "No metrics available"}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0">
          <Command>
            <CommandInput
              placeholder="Search metrics..."
              value={metricInputValue}
              onValueChange={setMetricInputValue}
              className="h-9"
            />
            <CommandList>
              <CommandEmpty className="px-4 py-2 text-sm">
                No metrics found.
              </CommandEmpty>
              <CommandGroup heading="Metrics">
                {filteredMetrics.map(([metricName, metricConfig]) => (
                  <CommandItem
                    key={metricName}
                    value={metricName}
                    onSelect={() => {
                      onMetricChange(metricName);
                      setMetricInputValue(""); // Clear input on selection
                      setMetricPopoverOpen(false); // Close popover
                    }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center">
                      <Check
                        className={clsx(
                          "mr-2 h-4 w-4",
                          selectedMetric === metricName
                            ? "opacity-100"
                            : "opacity-0",
                        )}
                      />
                      <span>{metricName}</span>
                    </div>
                    {metricConfig && (
                      <FeedbackBadges
                        metric={metricConfig}
                        showLevel={showLevelBadges}
                      />
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
