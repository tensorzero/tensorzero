import { useEffect, useState } from "react";
import { useRevalidator } from "react-router";

export default function AutoRefreshIndicator({
  isActive,
}: {
  isActive: boolean;
}) {
  if (!isActive) return null;

  return (
    <div className="flex items-center text-sm text-blue-600">
      <svg
        className="-ml-1 mr-2 h-4 w-4 animate-spin text-blue-600"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        ></circle>
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        ></path>
      </svg>
      Auto-refreshing
    </div>
  );
}

export const useAutoRefresh = (mostRecentDate: Date, interval = 1000) => {
  const [isAutoRefreshing, setIsAutoRefreshing] = useState(false);
  const revalidator = useRevalidator();

  useEffect(() => {
    // Only auto-refresh if it's been less than 1 minute since the most recent date
    const now = new Date();
    const differenceInMs = now.getTime() - mostRecentDate.getTime();
    const differenceInMinutes = differenceInMs / (1000 * 60);

    if (differenceInMinutes >= 1) {
      setIsAutoRefreshing(false);
      return; // Don't set up auto-refresh if data is older than 1 minute
    }

    setIsAutoRefreshing(true);

    // Set up interval for revalidation
    const intervalId = window.setInterval(() => {
      // Only revalidate if not already revalidating
      if (revalidator.state === "idle") {
        revalidator.revalidate();
      }
    }, interval);

    return () => {
      clearInterval(intervalId);
      setIsAutoRefreshing(false);
    };
  }, [mostRecentDate, revalidator, interval]);

  return isAutoRefreshing;
};
