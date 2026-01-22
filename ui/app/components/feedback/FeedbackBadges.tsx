import { Badge } from "~/components/ui/badge";
import type { FeedbackRow } from "~/types/tensorzero";
import type { FeedbackConfig } from "~/utils/config/feedback";

// Badge styles using orange/amber/yellow spectrum for visual cohesion
export const getBadgeStyle = (
  property: "type" | "optimize" | "level",
  value: string | undefined,
) => {
  switch (property) {
    case "type":
      switch (value) {
        case "boolean":
          return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300";
        case "float":
          return "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300";
        case "demonstration":
          return "bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-300";
        case undefined:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
        default:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
      }

    case "optimize":
      if (!value) return "";
      return value === "max"
        ? "bg-orange-200 text-orange-800 dark:bg-orange-800 dark:text-orange-200"
        : "bg-amber-200 text-amber-700 dark:bg-amber-800 dark:text-amber-200";

    case "level":
      return value === "episode"
        ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
        : "bg-yellow-200 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200";

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
