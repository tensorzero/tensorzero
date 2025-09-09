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
