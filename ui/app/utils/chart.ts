/**
 * Standard chart colors for consistent theming across all charts
 * Uses CSS custom properties defined in the theme
 */
export const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

/**
 * Standard chart margin for consistent spacing
 */
export const CHART_MARGIN = { top: 12, right: 0, bottom: 0, left: 0 } as const;

/**
 * Standard axis line stroke color
 */
export const CHART_AXIS_STROKE = "#e5e5e5";

/**
 * Format numbers for chart axes to avoid overflow with large numbers
 * Uses compact notation (K, M, B) for readability
 */
export function formatChartNumber(value: number): string {
  if (value === 0) return "0";

  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  if (abs >= 1_000_000_000) {
    return `${sign}${(abs / 1_000_000_000).toFixed(1).replace(/\.0$/, "")}B`;
  }
  if (abs >= 1_000_000) {
    return `${sign}${(abs / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (abs >= 1_000) {
    return `${sign}${(abs / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  }

  return value.toString();
}

/**
 * Format numbers for detailed display (tooltips, tables, etc.)
 * Uses full number with locale-specific thousand separators
 */
export function formatDetailedNumber(value: number): string {
  return value.toLocaleString();
}

/**
 * Format numbers with 3 significant digits and compact notation
 * Examples: 0.12, 1.23, 12.3, 123, 1.23k, 12.3k, 123k, 1.23M
 */
export function formatCompactNumber(value: number): string {
  if (value === 0) return "0";

  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  if (abs >= 1_000_000_000_000) {
    const n = abs / 1_000_000_000_000;
    return `${sign}${n >= 100 ? Math.round(n) : n >= 10 ? n.toFixed(1) : n.toFixed(2)}T`;
  }
  if (abs >= 1_000_000_000) {
    const n = abs / 1_000_000_000;
    return `${sign}${n >= 100 ? Math.round(n) : n >= 10 ? n.toFixed(1) : n.toFixed(2)}B`;
  }
  if (abs >= 1_000_000) {
    const n = abs / 1_000_000;
    return `${sign}${n >= 100 ? Math.round(n) : n >= 10 ? n.toFixed(1) : n.toFixed(2)}M`;
  }
  if (abs >= 1_000) {
    const n = abs / 1_000;
    return `${sign}${n >= 100 ? Math.round(n) : n >= 10 ? n.toFixed(1) : n.toFixed(2)}k`;
  }

  // For numbers < 1000, use 3 significant digits
  if (abs >= 100) {
    return `${sign}${Math.round(abs)}`;
  }
  if (abs >= 10) {
    return `${sign}${abs.toFixed(1).replace(/\.0$/, "")}`;
  }
  if (abs >= 1) {
    return `${sign}${abs.toFixed(2).replace(/\.?0+$/, "")}`;
  }
  // For decimals < 1, show up to 2 decimal places
  return `${sign}${abs.toFixed(2).replace(/\.?0+$/, "")}`;
}

/**
 * Format latency values for chart axes
 * Converts to appropriate time units (ms or s) for readability
 */
export function formatLatency(ms: number): string {
  if (ms === 0) return "0";

  const abs = Math.abs(ms);

  if (abs >= 1000) {
    const seconds = abs / 1000;
    if (seconds >= 100) {
      return `${Math.round(seconds)}s`;
    }
    if (seconds >= 10) {
      return `${seconds.toFixed(1).replace(/\.0$/, "")}s`;
    }
    return `${seconds.toFixed(2).replace(/\.?0+$/, "")}s`;
  }

  // For ms values, keep it simple
  if (abs >= 10) {
    return `${Math.round(abs)}ms`;
  }
  return `${abs.toFixed(1).replace(/\.0$/, "")}ms`;
}

/**
 * Helper to pad numbers with leading zeros
 */
function pad(num: number, size: number = 2): string {
  return num.toString().padStart(size, "0");
}

/**
 * Format timestamp for x-axis ticks based on time granularity
 * Uses local timezone and concise format (omits year for space)
 */
export function formatXAxisTimestamp(
  date: Date,
  granularity: "minute" | "hour" | "day" | "week" | "month" | "cumulative",
): string {
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());

  switch (granularity) {
    case "minute":
      // Format: MM-DD HH:mm
      return `${month}-${day} ${hours}:${minutes}`;
    case "hour":
      // Format: MM-DD HH:00
      return `${month}-${day} ${hours}:00`;
    case "day":
    case "week":
    case "month":
    case "cumulative":
      // Format: YYYY-MM-DD
      return `${date.getFullYear()}-${month}-${day}`;
  }
}

/**
 * Format timestamp for tooltips based on time granularity
 * Uses local timezone and verbose format (includes full date)
 */
export function formatTooltipTimestamp(
  date: Date,
  granularity: "minute" | "hour" | "day" | "week" | "month" | "cumulative",
): string {
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());

  switch (granularity) {
    case "minute":
      // Format: YYYY-MM-DD HH:mm
      return `${year}-${month}-${day} ${hours}:${minutes}`;
    case "hour":
      // Format: YYYY-MM-DD HH:00
      return `${year}-${month}-${day} ${hours}:00`;
    case "day":
    case "week":
    case "month":
    case "cumulative":
      // Format: YYYY-MM-DD
      return `${year}-${month}-${day}`;
  }
}
