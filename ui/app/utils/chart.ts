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
