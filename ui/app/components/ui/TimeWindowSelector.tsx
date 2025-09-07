import type { TimeWindow } from "tensorzero-node";
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
        <SelectItem value="hour">Hour</SelectItem>
        <SelectItem value="day">Day</SelectItem>
        <SelectItem value="week">Week</SelectItem>
        <SelectItem value="month">Month</SelectItem>
        <SelectItem value="cumulative">Cumulative</SelectItem>
      </SelectContent>
    </Select>
  );
}
