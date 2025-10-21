import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
import type { TimeWindow } from "tensorzero-node";

type TimeGranularitySelectorProps = {
  time_granularity: TimeWindow;
  onTimeGranularityChange: (time_granularity: TimeWindow) => void;
  includeCumulative?: boolean;
  includeMinute?: boolean;
  includeHour?: boolean;
};

export function TimeGranularitySelector({
  time_granularity: timeGranularity,
  onTimeGranularityChange,
  includeCumulative = true,
  includeMinute = false,
  includeHour = false,
}: TimeGranularitySelectorProps) {
  return (
    <div className="flex flex-col justify-center">
      <Select value={timeGranularity} onValueChange={onTimeGranularityChange}>
        <SelectTrigger>
          <SelectValue placeholder="Choose time granularity" />
        </SelectTrigger>
        <SelectContent>
          {includeMinute && <SelectItem value="minute">By Minute</SelectItem>}
          {includeHour && <SelectItem value="hour">Hourly</SelectItem>}
          <SelectItem value="day">Daily</SelectItem>
          <SelectItem value="week">Weekly</SelectItem>
          <SelectItem value="month">Monthly</SelectItem>
          {includeCumulative && (
            <SelectItem value="cumulative">All Time</SelectItem>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
