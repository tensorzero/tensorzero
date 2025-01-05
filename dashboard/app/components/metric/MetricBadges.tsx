import { Badge } from "~/components/ui/badge";
import type { MetricConfig } from "~/utils/config/metric";

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
        default:
          return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
      }

    case "optimize":
      if (!value) return "";
      return value === "max"
        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300";

    case "level":
      return value === "episode"
        ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300"
        : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-300";

    default:
      return "";
  }
};

type MetricBadgesProps = {
  metric: MetricConfig;
};

export function MetricBadges({ metric }: MetricBadgesProps) {
  return (
    <div className="flex gap-1.5">
      {/* Type badge */}
      <Badge className={getBadgeStyle("type", metric.type)}>
        {metric.type}
      </Badge>

      {/* Only show optimize badge if it's defined */}
      {"optimize" in metric && metric.optimize && (
        <Badge className={getBadgeStyle("optimize", metric.optimize)}>
          {metric.optimize}
        </Badge>
      )}

      {/* Level badge */}
      <Badge className={getBadgeStyle("level", metric.level)}>
        {metric.level}
      </Badge>
    </div>
  );
}
