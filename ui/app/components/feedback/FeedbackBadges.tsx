import { Badge } from "~/components/ui/badge";
import type { FeedbackRow } from "~/types/tensorzero";
import type { FeedbackConfig } from "~/utils/config/feedback";

// Move the getBadgeStyle function from MetricSelector
const getBadgeStyle = (
  property: "type" | "optimize" | "level",
  value: string | undefined,
) => {
  switch (property) {
    case "type":
      switch (value) {
        case "boolean":
          return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300";
        case "float":
          return "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300";
        case "demonstration":
          return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300";
        case undefined:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
        default:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
      }

    case "optimize":
      if (!value) return "";
      return value === "max"
        ? "bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-900 dark:text-fuchsia-300"
        : "bg-sky-100 text-sky-800 dark:bg-sky-900 dark:text-sky-300";

    case "level":
      return value === "episode"
        ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300"
        : "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-300";

    default:
      return "";
  }
};

type FeedbackBadgesProps = {
  metric: FeedbackConfig;
  row?: FeedbackRow;
  showLevel?: boolean;
};

export default function FeedbackBadges({
  metric,
  row,
  showLevel = true,
}: FeedbackBadgesProps) {
  if (!metric) return null;
  return (
    <div className="flex gap-1.5">
      {/* Type badge */}
      {metric.type !== "comment" && metric.type !== "demonstration" && (
        <Badge className={getBadgeStyle("type", metric.type)}>
          {metric.type}
        </Badge>
      )}

      {/* Only show optimize badge if it's defined */}
      {"optimize" in metric && metric.optimize && (
        <Badge className={getBadgeStyle("optimize", metric.optimize)}>
          {metric.optimize}
        </Badge>
      )}

      {/* Level badge */}
      {/* If the metric is a comment take the level from the row if available */}
      {showLevel &&
        metric.type === "comment" &&
        row?.type === "comment" &&
        row?.target_type && (
          <Badge className={getBadgeStyle("level", row.target_type)}>
            {row.target_type}
          </Badge>
        )}
      {showLevel && metric.type !== "comment" && (
        <Badge className={getBadgeStyle("level", metric.level)}>
          {metric.level}
        </Badge>
      )}
    </div>
  );
}
