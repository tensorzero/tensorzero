import { getTimestampTooltipData } from "~/utils/date";

interface TimestampTooltipProps {
  timestamp: string | number | Date;
}

export function TimestampTooltip({ timestamp }: TimestampTooltipProps) {
  const { formattedDate, formattedTime, relativeTime } =
    getTimestampTooltipData(timestamp);

  return (
    <div className="flex flex-col gap-1">
      <div>{formattedDate}</div>
      <div>{formattedTime}</div>
      <div>{relativeTime}</div>
    </div>
  );
}
