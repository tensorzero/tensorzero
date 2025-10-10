import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
import type { TimeWindowUnit } from "~/utils/clickhouse/function";

type TimeGranularitySelectorProps = {
  time_granularity: TimeWindowUnit;
  onTimeGranularityChange: (time_granularity: TimeWindowUnit) => void;
  includeCumulative?: boolean;
};

export function TimeGranularitySelector({
  time_granularity: timeGranularity,
  onTimeGranularityChange,
  includeCumulative = true,
}: TimeGranularitySelectorProps) {
  return (
    <div className="flex flex-col justify-center">
      <Select value={timeGranularity} onValueChange={onTimeGranularityChange}>
        <SelectTrigger>
          <SelectValue placeholder="Choose time granularity" />
        </SelectTrigger>
        <SelectContent>
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
