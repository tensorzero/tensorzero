import type { MetricsWithFeedbackResponse } from "~/types/tensorzero";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import { useConfig } from "~/context/config";

type MetricSelectorProps = {
  metricsWithFeedback: MetricsWithFeedbackResponse;
  selectedMetric: string;
  onMetricChange: (metric: string) => void;
};

export function MetricSelector({
  metricsWithFeedback,
  selectedMetric,
  onMetricChange,
}: MetricSelectorProps) {
  const config = useConfig();
  if (!metricsWithFeedback.metrics?.length) {
    return (
      <div className="text-fg-muted flex flex-col justify-center text-sm">
        No metrics available.
      </div>
    );
  }

  return (
    <div className="flex flex-col justify-center">
      <Select value={selectedMetric} onValueChange={onMetricChange}>
        <SelectTrigger>
          <SelectValue placeholder="Choose a metric" />
        </SelectTrigger>
        <SelectContent>
          {metricsWithFeedback.metrics.map((metric) => (
            <SelectItem key={metric.metric_name} value={metric.metric_name}>
              <div className="flex items-center justify-between">
                <span className="mr-2">{metric.metric_name}</span>
                {config?.metrics[metric.metric_name] && (
                  <FeedbackBadges
                    metric={config.metrics[metric.metric_name]!}
                  />
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
