export const formatDate = (date: Date) => {
  const formattedDate = date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const formattedTime = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }).format(date);

  return `${formattedDate} · ${formattedTime}`;
};

/**
 * Format date with time including seconds
 */
export const formatDateWithSeconds = (date: Date) => {
  const formattedDate = date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const formattedTime = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }).format(date);

  return `${formattedDate} · ${formattedTime}`;
};

/**
 * Get relative time string (e.g. "2 hours ago")
 */
export const getRelativeTimeString = (date: Date) => {
  const diff = (date.getTime() - new Date().getTime()) / 1000;
  const absDiff = Math.abs(diff);

  if (absDiff < 60) {
    // < 1 minute: display in seconds
    return new Intl.RelativeTimeFormat("en-US", {
      style: "long",
    }).format(Math.floor(diff), "second");
  } else if (absDiff < 60 * 60) {
    // < 1 hour: display in minutes
    return new Intl.RelativeTimeFormat("en-US", {
      style: "long",
    }).format(Math.floor(diff / 60), "minute");
  } else if (absDiff < 60 * 60 * 24) {
    // < 1 day: display in hours
    return new Intl.RelativeTimeFormat("en-US", {
      style: "long",
    }).format(Math.floor(diff / (60 * 60)), "hour");
  } else if (absDiff < 60 * 60 * 24 * 30) {
    // < 1 month: display in days
    return new Intl.RelativeTimeFormat("en-US", {
      style: "long",
    }).format(Math.floor(diff / (60 * 60 * 24)), "day");
  } else {
    // > 1 month: display in weeks
    return new Intl.RelativeTimeFormat("en-US", {
      style: "long",
    }).format(Math.floor(diff / (60 * 60 * 24 * 7)), "week");
  }
};

/**
 * Get timestamp data for tooltip
 */
export const getTimestampTooltipData = (timestamp: string | number | Date) => {
  const date = new Date(timestamp);

  return {
    // "Monday, January 20, 2025"
    formattedDate: date.toLocaleDateString("en-US", {
      dateStyle: "full",
    }),
    // "1:23:45 PM EST"
    formattedTime: date.toLocaleTimeString("en-US", {
      timeStyle: "long",
    }),
    // "2 hours ago"
    relativeTime: getRelativeTimeString(date),
  };
};

/**
 * Time window units for time series granularity
 */
export type TimeWindow =
  | "minute"
  | "hour"
  | "day"
  | "week"
  | "month"
  | "cumulative";

/**
 * Normalize a date to match a specific time period granularity by truncating
 * to the start of the period.
 *
 * @param date - The date to normalize
 * @param granularity - The time window unit to normalize to
 * @returns A new Date object truncated to the start of the period
 *
 * @example
 * // Normalize to start of day
 * normalizePeriod(new Date("2025-01-15T14:30:00Z"), "day")
 * // Returns: 2025-01-15T00:00:00Z
 */
export function normalizePeriod(date: Date, granularity: TimeWindow): Date {
  const normalized = new Date(date);
  switch (granularity) {
    case "minute":
      // Truncate to minute
      normalized.setUTCSeconds(0, 0);
      break;
    case "hour":
      // Truncate to hour
      normalized.setUTCMinutes(0, 0, 0);
      break;
    case "day":
    case "week":
    case "month":
      // Truncate to day
      normalized.setUTCHours(0, 0, 0, 0);
      break;
    case "cumulative":
      // No truncation needed for cumulative
      break;
  }
  return normalized;
}

/**
 * Add one period to a date based on the specified granularity.
 *
 * @param date - The starting date
 * @param granularity - The time window unit to add
 * @returns ISO string of the date plus one period, normalized to the period boundary
 *
 * @example
 * // Add one day
 * addPeriod(new Date("2025-01-15T00:00:00Z"), "day")
 * // Returns: "2025-01-16T00:00:00.000Z"
 *
 * @example
 * // Add one week
 * addPeriod(new Date("2025-01-15T00:00:00Z"), "week")
 * // Returns: "2025-01-22T00:00:00.000Z"
 */
export function addPeriod(date: Date, granularity: TimeWindow): string {
  const result = new Date(date);
  switch (granularity) {
    case "minute":
      result.setUTCMinutes(result.getUTCMinutes() + 1);
      break;
    case "hour":
      result.setUTCHours(result.getUTCHours() + 1);
      break;
    case "day":
      result.setUTCDate(result.getUTCDate() + 1);
      break;
    case "week":
      result.setUTCDate(result.getUTCDate() + 7);
      break;
    case "month":
      result.setUTCMonth(result.getUTCMonth() + 1);
      break;
    case "cumulative":
      // No period addition for cumulative
      break;
  }
  return normalizePeriod(result, granularity).toISOString();
}
