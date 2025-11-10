import type { TimeWindow } from "~/types/tensorzero";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";

export function TimeWindowSelector({
  value,
  onValueChange,
  className,
}: {
  value: TimeWindow;
  onValueChange: (timeWindow: TimeWindow) => void;
  className?: string;
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger className={className}>
        <SelectValue placeholder="Choose time granularity" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="hour">Hourly</SelectItem>
        <SelectItem value="day">Daily</SelectItem>
        <SelectItem value="week">Weekly</SelectItem>
        <SelectItem value="month">Monthly</SelectItem>
        <SelectItem value="cumulative">All Time</SelectItem>
      </SelectContent>
    </Select>
  );
}
