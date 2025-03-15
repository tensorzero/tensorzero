export const formatDate = (date: Date) => {
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

/**
 * Format date with time including seconds
 */
export const formatDateWithSeconds = (date: Date) => {
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    second: "numeric",
    hour12: true,
  };

  return new Date(date).toLocaleString("en-US", options);
};

/**
 * Get relative time string (e.g. "2 hours ago")
 */
export const getRelativeTimeString = (date: Date) => {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  const diffMonths = Math.floor(diffDays / 30);

  if (diffMonths > 0) {
    return `${diffMonths} month${diffMonths > 1 ? "s" : ""}, ${diffDays % 30} day${diffDays % 30 !== 1 ? "s" : ""}, ${diffHours % 24} hour${diffHours % 24 !== 1 ? "s" : ""} ago`;
  } else if (diffDays > 0) {
    return `${diffDays} day${diffDays > 1 ? "s" : ""}, ${diffHours % 24} hour${diffHours % 24 !== 1 ? "s" : ""} ago`;
  } else if (diffHours > 0) {
    return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
  } else {
    return `${diffMinutes} minute${diffMinutes !== 1 ? "s" : ""} ago`;
  }
};

/**
 * Format date as MM/DD/YYYY
 */
export const formatDateShort = (date: Date) => {
  return date.toLocaleDateString("en-US", {
    month: "2-digit",
    day: "2-digit",
    year: "numeric",
  });
};

/**
 * Format time as H:MM:SS AM/PM with user's local timezone
 */
export const formatTimeWithTimezone = (date: Date) => {
  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
    timeZoneName: "short",
  });
};

/**
 * Get timestamp data for tooltip
 */
export const getTimestampTooltipData = (timestamp: string | number | Date) => {
  const date = new Date(timestamp);

  return {
    formattedDate: formatDateShort(date),
    formattedTime: formatTimeWithTimezone(date),
    relativeTime: getRelativeTimeString(date),
  };
};
