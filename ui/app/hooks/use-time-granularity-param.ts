import { useNavigate, useSearchParams } from "react-router";
import type { TimeWindow } from "~/types/tensorzero";

/**
 * A hook to manage a time granularity URL parameter
 * @param paramName - The name of the URL parameter to manage
 * @param defaultValue - The default time granularity (defaults to "week")
 * @returns [timeGranularity, setTimeGranularity] - current value and setter
 */
export function useTimeGranularityParam(
  paramName: string,
  defaultValue: TimeWindow = "week",
): [TimeWindow, (granularity: TimeWindow) => void] {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const timeGranularity =
    (searchParams.get(paramName) as TimeWindow) || defaultValue;

  const setTimeGranularity = (granularity: TimeWindow) => {
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set(paramName, granularity);
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return [timeGranularity, setTimeGranularity];
}
