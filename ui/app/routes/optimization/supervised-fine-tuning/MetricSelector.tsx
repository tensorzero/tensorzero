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
import { MetricBadges } from "~/components/metric/MetricBadges";

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
          <div className="grid gap-x-8 md:grid-cols-2">
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
                    <div className="flex w-full items-center justify-between">
                      <span>None</span>
                    </div>
                  </SelectItem>
                  {Object.entries(config.metrics).map(([name, metric]) => (
                    <SelectItem key={name} value={name}>
                      <div className="flex w-full items-center justify-between">
                        <span>{name}</span>
                        <div className="ml-2">
                          <MetricBadges metric={metric} />
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
                    <div className="rounded-lg bg-gray-100 p-4">
                      <FormLabel>Threshold</FormLabel>
                      <Input
                        type="number"
                        step="0.01"
                        {...thresholdField}
                        className="border-none bg-transparent focus:ring-0"
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
