/**
 * Format a dollar amount with trailing zeros trimmed but always at least 2 decimal places.
 *
 * Examples:
 *   formatCost(0.0042)   → "$0.0042"
 *   formatCost(0)        → "$0.00"
 *   formatCost(1.5)      → "$1.50"
 */
export function formatCost(cost: number): string {
  const fixed = cost.toFixed(9);
  // Trim trailing zeros but keep at least 2 decimal places
  const trimmed = fixed.replace(/0+$/, "");
  const [integer, decimal = ""] = trimmed.split(".");
  const paddedDecimal = decimal.padEnd(2, "0");
  return `$${integer}.${paddedDecimal}`;
}
