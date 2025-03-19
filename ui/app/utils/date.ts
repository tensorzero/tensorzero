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
