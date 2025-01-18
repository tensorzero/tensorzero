import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
// import { Badge } from "~/components/ui/badge";
// import { MetricBadges } from "~/components/metric/MetricBadges";
import React from "react";

type MetricSelectorProps = {
  metricsWithFeedback: MetricsWithFeedbackData;
  selectedMetric: string;
  onMetricChange: (metric: string) => void;
};

export function MetricSelector({
  metricsWithFeedback,
  selectedMetric,
  onMetricChange,
}: MetricSelectorProps) {
  if (!metricsWithFeedback.metrics?.length) {
    return (
      <div className="flex flex-col justify-center">
        <label className="mb-2">Metric</label>
        <div>No metrics available</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col justify-center">
      <label className="mb-2">Metric</label>
      <Select value={selectedMetric} onValueChange={onMetricChange}>
        <SelectTrigger>
          <SelectValue placeholder="Choose a metric" />
        </SelectTrigger>
        <SelectContent>
          {metricsWithFeedback.metrics.map((metric) => (
            <SelectItem key={metric.metric_name} value={metric.metric_name}>
              {metric.metric_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
