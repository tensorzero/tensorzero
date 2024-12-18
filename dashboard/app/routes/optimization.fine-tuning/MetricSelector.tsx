import { Control } from "react-hook-form";
import { SFTFormValues } from "./types";
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

type MetricSelectorProps = {
  control: Control<SFTFormValues>;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
  config: Config;
  onMetricChange: (value: string) => void;
};

export function MetricSelector({
  control,
  feedbackCount,
  curatedInferenceCount,
  config,
  onMetricChange,
}: MetricSelectorProps) {
  return (
    <FormField
      control={control}
      name="metric"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Metric</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Select
              onValueChange={(value) => {
                field.onChange(value);
                onMetricChange(value);
              }}
              value={field.value}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a metric" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(config.metrics).map(([name, metric]) => (
                  <SelectItem key={name} value={name}>
                    <div className="flex items-center justify-between w-full">
                      <span>{name}</span>
                      <div className="ml-2 flex gap-1.5">
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                            ${
                              metric.type === "boolean"
                                ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
                                : "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300"
                            }`}
                        >
                          {metric.type}
                        </span>
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                            ${
                              metric.optimize === "max"
                                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
                                : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
                            }`}
                        >
                          {metric.optimize}
                        </span>
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                            ${
                              metric.level === "episode"
                                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300"
                                : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-300"
                            }`}
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
                      min={0}
                      max={1}
                      {...thresholdField}
                      className="bg-transparent border-none focus:ring-0"
                      onChange={(e) =>
                        thresholdField.onChange(Number(e.target.value))
                      }
                    />
                  </div>
                )}
              />
            )}

            <div className="space-y-1 text-sm text-muted-foreground">
              <div>
                Feedbacks:{" "}
                {field.value ? (
                  <span className="font-medium">{feedbackCount}</span>
                ) : (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                )}
              </div>
              <div>
                Curated Inferences:{" "}
                {field.value ? (
                  <span className="font-medium">{curatedInferenceCount}</span>
                ) : (
                  <Skeleton className="inline-block h-4 w-16 align-middle" />
                )}
              </div>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
