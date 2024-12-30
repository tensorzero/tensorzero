import type { Control } from "react-hook-form";
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

// Get badge styling based on metric properties
const getBadgeStyle = (
  property: "type" | "optimize" | "level",
  value: string | undefined,
) => {
  switch (property) {
    // Type badges
    case "type":
      switch (value) {
        case "boolean":
          return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300";
        case "float":
          return "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300";
        case "demonstration":
          return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300";
        default:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
      }

    // Optimization direction badges
    case "optimize":
      if (!value) return ""; // Don't render badge if optimize is undefined
      return value === "max"
        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300";

    // Level badges
    case "level":
      return value === "episode"
        ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300"
        : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-300";

    default:
      return "";
  }
};

type MetricSelectorProps = {
  control: Control<SFTFormValues>;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  config: Config;
  onMetricChange: (value: string | null) => void;
  onThresholdChange: (value: number) => void;
};

export function MetricSelector({
  control,
  feedbackCount,
  curatedInferenceCount,
  config,
  onMetricChange,
  onThresholdChange,
}: MetricSelectorProps) {
  return (
    <FormField
      control={control}
      name="metric"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Metric</FormLabel>
          <div className="grid md:grid-cols-2 gap-x-8">
            <div className="space-y-2">
              <Select
                onValueChange={(value: string) => {
                  const metricValue = value === "none" ? null : value;
                  field.onChange(metricValue);
                  onMetricChange(metricValue);
                }}
                value={field.value ?? "none"}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a metric" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">
                    <div className="flex items-center justify-between w-full">
                      <span>None</span>
                    </div>
                  </SelectItem>
                  {Object.entries(config.metrics).map(([name, metric]) => (
                    <SelectItem key={name} value={name}>
                      <div className="flex items-center justify-between w-full">
                        <span>{name}</span>
                        <div className="ml-2 flex gap-1.5">
                          {/* Type badge */}
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                              ${getBadgeStyle("type", metric.type)}`}
                          >
                            {metric.type}
                          </span>
                          {/* Only show optimize badge if it's defined */}
                          {"optimize" in metric && metric.optimize && (
                            <span
                              className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                ${getBadgeStyle("optimize", metric.optimize)}`}
                            >
                              {metric.optimize}
                            </span>
                          )}
                          {/* Level badge */}
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                              ${getBadgeStyle("level", metric.level)}`}
                          >
                            {metric.level}
                          </span>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {field.value && config.metrics[field.value]?.type === "float" && (
                <FormField
                  control={control}
                  name="threshold"
                  render={({ field: thresholdField }) => (
                    <div className="p-4 bg-gray-100 rounded-lg">
                      <FormLabel>Threshold</FormLabel>
                      <Input
                        type="number"
                        step="0.01"
                        {...thresholdField}
                        className="bg-transparent border-none focus:ring-0"
                        onChange={(e) => {
                          thresholdField.onChange(Number(e.target.value));
                          onThresholdChange(Number(e.target.value));
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
