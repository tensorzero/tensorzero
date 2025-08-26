import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
import type { TimeWindowUnit } from "~/utils/clickhouse/function";

type TimeGranularitySelectorProps = {
  timeGranularity: TimeWindowUnit;
  onTimeGranularityChange: (time_granularity: TimeWindowUnit) => void;
};

export function TimeGranularitySelector({
  timeGranularity,
  onTimeGranularityChange,
}: TimeGranularitySelectorProps) {
  return (
    <div className="flex flex-col justify-center">
      <Select value={timeGranularity} onValueChange={onTimeGranularityChange}>
        <SelectTrigger>
          <SelectValue placeholder="Choose time granularity" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="day">Day</SelectItem>
          <SelectItem value="week">Week</SelectItem>
          <SelectItem value="month">Month</SelectItem>
          <SelectItem value="cumulative">Cumulative</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
