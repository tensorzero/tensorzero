/**
 * Base chart colors from CSS custom properties.
 * For more than 5 series, use `getChartColor(index)` which generates
 * unlimited distinct colors via golden-angle hue rotation.
 */
export const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

/**
 * Get a chart color by index. Returns themed CSS colors for the first 5,
 * then generates additional distinct colors using the golden angle.
 */
export function getChartColor(index: number): string {
  if (index < CHART_COLORS.length) {
    return CHART_COLORS[index];
  }
  // Golden angle (~137.5°) produces maximally spaced hues
  const hue = ((index - CHART_COLORS.length) * 137.508 + 60) % 360;
  return `hsl(${hue.toFixed(0)} 55% 55%)`;
}

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
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());

  const MONTH_SHORT = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  switch (granularity) {
    case "minute":
      // Format: Mar 30 14:05
      return `${MONTH_SHORT[date.getMonth()]} ${date.getDate()} ${hours}:${minutes}`;
    case "hour":
      // Format: Mar 30 14:00
      return `${MONTH_SHORT[date.getMonth()]} ${date.getDate()} ${hours}:00`;
    case "day":
      // Format: Mar 30
      return `${MONTH_SHORT[date.getMonth()]} ${date.getDate()}`;
    case "week":
      // Format: Mar 29 (week start date, like Grafana/Datadog)
      return `${MONTH_SHORT[date.getMonth()]} ${date.getDate()}`;
    case "month":
      // Format: Mar 2026
      return `${MONTH_SHORT[date.getMonth()]} ${date.getFullYear()}`;
    case "cumulative":
      return "All time";
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
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());

  const MONTH_LONG = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  switch (granularity) {
    case "minute":
      // Format: March 30, 2026 14:05
      return `${MONTH_LONG[date.getMonth()]} ${date.getDate()}, ${year} ${hours}:${minutes}`;
    case "hour":
      // Format: March 30, 2026 14:00
      return `${MONTH_LONG[date.getMonth()]} ${date.getDate()}, ${year} ${hours}:00`;
    case "day":
      // Format: March 30, 2026
      return `${MONTH_LONG[date.getMonth()]} ${date.getDate()}, ${year}`;
    case "week":
      // Format: Week of March 29, 2026
      return `Week of ${MONTH_LONG[date.getMonth()]} ${date.getDate()}, ${year}`;
    case "month":
      // Format: March 2026
      return `${MONTH_LONG[date.getMonth()]} ${year}`;
    case "cumulative":
      return "All time";
  }
}
