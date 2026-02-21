import { Skeleton } from "~/components/ui/skeleton";

// Default bar heights for skeleton (percentage of container height)
const SKELETON_BAR_HEIGHTS = [45, 65, 35, 80, 55, 40, 70, 50, 60, 30];

/**
 * Generic bar chart skeleton with Y-axis, grid lines, bars, X-axis, and legend.
 * Matches the visual structure of Recharts BarChart.
 */
export function BarChartSkeleton({ className }: { className?: string }) {
  return (
    <div className={className}>
      {/* Chart container - h-80 matches ChartContainer in production */}
      <div className="flex h-80 w-full flex-col">
        {/* Chart area with Y-axis and bars */}
        <div className="flex flex-1">
          {/* Y-axis labels - w-[60px] matches Recharts default YAxis width */}
          <div className="flex w-[60px] flex-col items-end justify-between py-2 pr-2">
            <Skeleton className="h-3 w-8" />
            <Skeleton className="h-3 w-10" />
            <Skeleton className="h-3 w-6" />
            <Skeleton className="h-3 w-9" />
          </div>
          {/* Chart content area */}
          <div className="relative flex-1">
            {/* Horizontal grid lines */}
            <div className="absolute inset-0 flex flex-col justify-between py-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="border-border/40 w-full border-t" />
              ))}
            </div>
            {/* Bars */}
            <div className="absolute inset-x-0 top-2 bottom-2 flex items-end gap-2 px-2">
              {SKELETON_BAR_HEIGHTS.map((height, i) => (
                <Skeleton
                  key={i}
                  className="flex-1 rounded-t"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </div>
        </div>
        {/* X-axis labels */}
        <div className="ml-[60px] flex justify-between px-2 pt-2">
          <Skeleton className="h-3 w-8" />
          <Skeleton className="h-3 w-8" />
          <Skeleton className="h-3 w-8" />
          <Skeleton className="h-3 w-8" />
          <Skeleton className="h-3 w-8" />
        </div>
        {/* Legend - inside h-80 container, matches ChartLegendContent placement */}
        <div className="flex justify-center gap-4 pt-3">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-16" />
        </div>
      </div>
    </div>
  );
}
