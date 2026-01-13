import * as React from "react";
import * as RechartsPrimitive from "recharts";
import { useAsyncError } from "react-router";
import { AlertTriangle } from "lucide-react";

import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";
import { ChartErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";

// Format: { THEME_NAME: CSS_SELECTOR }
const THEMES = { light: "", dark: ".dark" } as const;

export type ChartConfig = {
  [k in string]: {
    label?: React.ReactNode;
    icon?: React.ComponentType;
  } & (
    | { color?: string; theme?: never }
    | { color?: never; theme: Record<keyof typeof THEMES, string> }
  );
};

type ChartContextProps = {
  config: ChartConfig;
};

const ChartContext = React.createContext<ChartContextProps | null>(null);

function useChart() {
  const context = React.useContext(ChartContext);

  if (!context) {
    throw new Error("useChart must be used within a <ChartContainer />");
  }

  return context;
}

const ChartContainer = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    config: ChartConfig;
    children: React.ComponentProps<
      typeof RechartsPrimitive.ResponsiveContainer
    >["children"];
  }
>(({ id, className, children, config, ...props }, ref) => {
  const uniqueId = React.useId();
  const chartId = `chart-${id || uniqueId.replace(/:/g, "")}`;

  return (
    <ChartContext.Provider value={{ config }}>
      <div
        data-chart={chartId}
        ref={ref}
        className={cn(
          "[&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line[stroke='#ccc']]:stroke-border/50 [&_.recharts-curve.recharts-tooltip-cursor]:stroke-border [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border [&_.recharts-radial-bar-background-sector]:fill-muted [&_.recharts-reference-line_[stroke='#ccc']]:stroke-border flex aspect-video justify-center text-xs [&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-layer]:outline-hidden [&_.recharts-rectangle.recharts-tooltip-cursor]:fill-orange-500/10 [&_.recharts-sector]:outline-hidden [&_.recharts-sector[stroke='#fff']]:stroke-transparent [&_.recharts-surface]:outline-hidden",
          className,
        )}
        {...props}
      >
        <ChartStyle id={chartId} config={config} />
        <RechartsPrimitive.ResponsiveContainer>
          {children}
        </RechartsPrimitive.ResponsiveContainer>
      </div>
    </ChartContext.Provider>
  );
});
ChartContainer.displayName = "Chart";

const ChartStyle = ({ id, config }: { id: string; config: ChartConfig }) => {
  const colorConfig = Object.entries(config).filter(
    ([, config]) => config.theme || config.color,
  );

  if (!colorConfig.length) {
    return null;
  }

  return (
    <style
      dangerouslySetInnerHTML={{
        __html: Object.entries(THEMES)
          .map(
            ([theme, prefix]) => `
${prefix} [data-chart=${id}] {
${colorConfig
  .map(([key, itemConfig]) => {
    const color =
      itemConfig.theme?.[theme as keyof typeof itemConfig.theme] ||
      itemConfig.color;
    return color ? `  --color-${key}: ${color};` : null;
  })
  .join("\n")}
}
`,
          )
          .join("\n"),
      }}
    />
  );
};

const ChartTooltip = RechartsPrimitive.Tooltip;

const ChartTooltipContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<typeof RechartsPrimitive.Tooltip> &
    React.ComponentProps<"div"> & {
      hideLabel?: boolean;
      hideIndicator?: boolean;
      indicator?: "line" | "dot" | "dashed";
      nameKey?: string;
      labelKey?: string;
    }
>(
  (
    {
      active,
      payload,
      className,
      indicator = "dot",
      hideLabel = false,
      hideIndicator = false,
      label,
      labelFormatter,
      labelClassName,
      formatter,
      color,
      nameKey,
      labelKey,
    },
    ref,
  ) => {
    const { config } = useChart();

    const tooltipLabel = React.useMemo(() => {
      if (hideLabel || !payload?.length) {
        return null;
      }

      const [item] = payload;
      const key = `${labelKey || item.dataKey || item.name || "value"}`;
      const itemConfig = getPayloadConfigFromPayload(config, item, key);
      const value =
        !labelKey && typeof label === "string"
          ? config[label as keyof typeof config]?.label || label
          : itemConfig?.label;

      if (labelFormatter) {
        return (
          <div className={cn("font-medium", labelClassName)}>
            {labelFormatter(value, payload)}
          </div>
        );
      }

      if (!value) {
        return null;
      }

      return <div className={cn("font-medium", labelClassName)}>{value}</div>;
    }, [
      label,
      labelFormatter,
      payload,
      hideLabel,
      labelClassName,
      config,
      labelKey,
    ]);

    if (!active || !payload?.length) {
      return null;
    }

    const nestLabel = payload.length === 1 && indicator !== "dot";

    return (
      <div
        ref={ref}
        className={cn(
          "border-border/50 bg-background grid min-w-[8rem] items-start gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs shadow-xl",
          className,
        )}
      >
        {!nestLabel ? tooltipLabel : null}
        <div className="grid gap-1.5">
          {payload.map((item, index) => {
            const key = `${nameKey || item.name || item.dataKey || "value"}`;
            const itemConfig = getPayloadConfigFromPayload(config, item, key);
            const indicatorColor = color || item.payload.fill || item.color;

            return (
              <div
                key={item.dataKey}
                className={cn(
                  "[&>svg]:text-muted-foreground flex w-full flex-wrap items-stretch gap-2 [&>svg]:h-2.5 [&>svg]:w-2.5",
                  indicator === "dot" && "items-center",
                )}
              >
                {formatter && item?.value !== undefined && item.name ? (
                  formatter(item.value, item.name, item, index, item.payload)
                ) : (
                  <>
                    {itemConfig?.icon ? (
                      <itemConfig.icon />
                    ) : (
                      !hideIndicator && (
                        <div
                          className={cn(
                            "shrink-0 rounded-[2px] border-(--color-border) bg-(--color-bg)",
                            {
                              "h-2.5 w-2.5": indicator === "dot",
                              "w-1": indicator === "line",
                              "w-0 border-[1.5px] border-dashed bg-transparent":
                                indicator === "dashed",
                              "my-0.5": nestLabel && indicator === "dashed",
                            },
                          )}
                          style={
                            {
                              "--color-bg": indicatorColor,
                              "--color-border": indicatorColor,
                            } as React.CSSProperties
                          }
                        />
                      )
                    )}
                    <div
                      className={cn(
                        "flex flex-1 justify-between leading-none",
                        nestLabel ? "items-end" : "items-center",
                      )}
                    >
                      <div className="grid gap-1.5">
                        {nestLabel ? tooltipLabel : null}
                        <span className="text-muted-foreground">
                          {itemConfig?.label || item.name}
                        </span>
                      </div>
                      {item.value && (
                        <span className="text-foreground font-mono font-medium tabular-nums">
                          {item.value.toLocaleString()}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  },
);
ChartTooltipContent.displayName = "ChartTooltip";

const ChartLegend = RechartsPrimitive.Legend;

const ChartLegendContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> &
    Pick<RechartsPrimitive.LegendProps, "payload" | "verticalAlign"> & {
      hideIcon?: boolean;
      nameKey?: string;
    }
>(
  (
    { className, hideIcon = false, payload, verticalAlign = "bottom", nameKey },
    ref,
  ) => {
    const { config } = useChart();

    if (!payload?.length) {
      return null;
    }

    return (
      <div
        ref={ref}
        className={cn(
          "flex flex-wrap items-center justify-center gap-4",
          verticalAlign === "top" ? "pb-3" : "pt-3",
          className,
        )}
      >
        {payload.map((item) => {
          const key = `${nameKey || item.dataKey || "value"}`;
          const itemConfig = getPayloadConfigFromPayload(config, item, key);

          return (
            <div
              key={item.value}
              className={cn(
                "[&>svg]:text-muted-foreground flex items-center gap-1.5 [&>svg]:h-3 [&>svg]:w-3",
              )}
            >
              {itemConfig?.icon && !hideIcon ? (
                <itemConfig.icon />
              ) : (
                <div
                  className="h-2 w-2 shrink-0 rounded-[2px]"
                  style={{
                    backgroundColor: item.color,
                  }}
                />
              )}
              {itemConfig?.label}
            </div>
          );
        })}
      </div>
    );
  },
);
ChartLegendContent.displayName = "ChartLegend";

// Helper to extract item config from a payload.
function getPayloadConfigFromPayload(
  config: ChartConfig,
  payload: unknown,
  key: string,
) {
  if (typeof payload !== "object" || payload === null) {
    return undefined;
  }

  const payloadPayload =
    "payload" in payload &&
    typeof payload.payload === "object" &&
    payload.payload !== null
      ? payload.payload
      : undefined;

  let configLabelKey: string = key;

  if (
    key in payload &&
    typeof payload[key as keyof typeof payload] === "string"
  ) {
    configLabelKey = payload[key as keyof typeof payload] as string;
  } else if (
    payloadPayload &&
    key in payloadPayload &&
    typeof payloadPayload[key as keyof typeof payloadPayload] === "string"
  ) {
    configLabelKey = payloadPayload[
      key as keyof typeof payloadPayload
    ] as string;
  }

  return configLabelKey in config
    ? config[configLabelKey]
    : config[key as keyof typeof config];
}

// Default bar heights for skeleton (percentage of container height)
const SKELETON_BAR_HEIGHTS = [45, 65, 35, 80, 55, 40, 70, 50, 60, 30];

/**
 * Generic bar chart skeleton with Y-axis, grid lines, bars, X-axis, and legend.
 * Matches the visual structure of Recharts BarChart.
 */
function BarChartSkeleton({ className }: { className?: string }) {
  return (
    <div className={className}>
      {/* Chart container - h-72 matches ChartContainer in production */}
      <div className="flex h-72 w-full flex-col">
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

/**
 * Generic line chart skeleton with Y-axis, grid lines, line, X-axis, and legend.
 * Matches the visual structure of Recharts LineChart.
 */
function LineChartSkeleton({ className }: { className?: string }) {
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

function ChartAsyncErrorState({
  defaultMessage = "Failed to load chart data",
}: {
  defaultMessage?: string;
}) {
  const error = useAsyncError();
  const message = error instanceof Error ? error.message : defaultMessage;

  return (
    <ChartErrorNotice
      icon={AlertTriangle}
      title="Chart Error"
      description={message}
    />
  );
}

/**
 * Standalone chart legend component for use outside of Recharts
 * Renders a simple flex-wrap legend with colored indicators
 */
function BasicChartLegend({
  items,
  colors,
}: {
  items: string[];
  colors: readonly string[];
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4 pt-6">
      {items.map((name, index) => (
        <div key={name} className="flex items-center gap-1.5">
          <div
            className="h-2 w-2 shrink-0 rounded-[2px]"
            style={{
              backgroundColor: colors[index % colors.length],
            }}
          />
          <span className="font-mono text-xs">{name}</span>
        </div>
      ))}
    </div>
  );
}

export {
  BarChartSkeleton,
  BasicChartLegend,
  ChartAsyncErrorState,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartStyle,
  ChartTooltip,
  ChartTooltipContent,
  LineChartSkeleton,
};
