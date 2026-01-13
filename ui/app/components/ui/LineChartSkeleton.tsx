import { Skeleton } from "~/components/ui/skeleton";

/**
 * Generic line chart skeleton with Y-axis, grid lines, distribution curves,
 * X-axis, and legend. Matches the visual structure of Recharts LineChart.
 *
 * Uses SVG for precise grid rendering with:
 * - Explicit chart boundaries (top, bottom, left, right)
 * - 11 evenly-spaced vertical grid lines
 * - 3 horizontal interior grid lines
 * - 3 distribution curve polylines
 */
export function LineChartSkeleton({ className }: { className?: string }) {
  // 11 evenly spaced vertical grid lines (not at edges)
  const verticalLines = Array.from({ length: 11 }, (_, i) =>
    Math.round((100 / 12) * (i + 1)),
  );

  return (
    <div className={className}>
      {/* Chart container - h-72 matches ChartContainer in production */}
      <div className="flex h-72 w-full flex-col">
        {/* Chart area with Y-axis and line */}
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
            {/* Grid and distribution lines via SVG */}
            <svg
              className="absolute inset-0 h-full w-full"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {/* Chart boundary - top */}
              <line
                x1={0}
                y1={0}
                x2={100}
                y2={0}
                stroke="currentColor"
                strokeWidth="1"
                className="text-border/40"
                vectorEffect="non-scaling-stroke"
              />
              {/* Chart boundary - bottom */}
              <line
                x1={0}
                y1={100}
                x2={100}
                y2={100}
                stroke="currentColor"
                strokeWidth="1"
                className="text-border/40"
                vectorEffect="non-scaling-stroke"
              />
              {/* Chart boundary - left */}
              <line
                x1={0}
                y1={0}
                x2={0}
                y2={100}
                stroke="currentColor"
                strokeWidth="1"
                className="text-border/40"
                vectorEffect="non-scaling-stroke"
              />
              {/* Chart boundary - right */}
              <line
                x1={100}
                y1={0}
                x2={100}
                y2={100}
                stroke="currentColor"
                strokeWidth="1"
                className="text-border/40"
                vectorEffect="non-scaling-stroke"
              />
              {/* Horizontal grid lines - interior only */}
              {[25, 50, 75].map((y) => (
                <line
                  key={`h-${y}`}
                  x1={0}
                  y1={y}
                  x2={100}
                  y2={y}
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-border/40"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {/* Vertical grid lines - 11 evenly spaced */}
              {verticalLines.map((x) => (
                <line
                  key={`v-${x}`}
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={100}
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-border/40"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {/* Distribution line 1 */}
              <polyline
                points="0,85 8,84 17,82 25,79 33,75 42,70 50,64 58,56 67,46 75,36 83,28 92,22 100,20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-muted-foreground/20"
                vectorEffect="non-scaling-stroke"
              />
              {/* Distribution line 2 */}
              <polyline
                points="0,88 8,87 17,85 25,82 33,78 42,73 50,67 58,59 67,50 75,41 83,34 92,28 100,26"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-muted-foreground/20"
                vectorEffect="non-scaling-stroke"
              />
              {/* Distribution line 3 */}
              <polyline
                points="0,92 8,91 17,89 25,86 33,82 42,77 50,71 58,63 67,54 75,46 83,40 92,35 100,33"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-muted-foreground/20"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>
        </div>
        {/* X-axis labels - positioned to align with vertical grid lines */}
        <div className="relative ml-[60px] h-5 pt-2">
          {/* Ticks at 0%, 25%, 50%, 75%, 100% (align with some vertical lines) */}
          <Skeleton
            className="absolute h-3 w-6"
            style={{ left: "0%", transform: "translateX(-50%)" }}
          />
          <Skeleton
            className="absolute h-3 w-6"
            style={{ left: "25%", transform: "translateX(-50%)" }}
          />
          <Skeleton
            className="absolute h-3 w-6"
            style={{ left: "50%", transform: "translateX(-50%)" }}
          />
          <Skeleton
            className="absolute h-3 w-6"
            style={{ left: "75%", transform: "translateX(-50%)" }}
          />
          <Skeleton
            className="absolute h-3 w-6"
            style={{ left: "100%", transform: "translateX(-50%)" }}
          />
        </div>
      </div>
      {/* Legend - outside h-72 container, matches BasicChartLegend placement */}
      <div className="flex justify-center gap-4 pt-6">
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-16" />
      </div>
    </div>
  );
}
