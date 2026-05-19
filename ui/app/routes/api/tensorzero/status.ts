import { useFetcher } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { useEffect, useEffectEvent, useMemo } from "react";
import { logger } from "~/utils/logger";
import type { StatusResponse } from "~/types/tensorzero";

/**
 * Loader that fetches the TensorZero Gateway status.
 */
export async function loader(): Promise<StatusResponse | undefined> {
  try {
    const status = await getTensorZeroClient().status();
    return status;
  } catch (error) {
    logger.error("Failed to fetch TensorZero status:", error);
    return undefined;
  }
}

/**
 * A hook that fetches the status of the TensorZero Gateway.
 */
export function useTensorZeroStatusFetcher() {
  const statusFetcher = useFetcher();
  const status = statusFetcher.data;
  const isLoading = statusFetcher.state === "loading";

  const fetcherEvent = useEffectEvent(() => {
    if (statusFetcher.state === "idle" && !statusFetcher.data) {
      statusFetcher.load("/api/tensorzero/status");
    }
  });
  useEffect(() => {
    fetcherEvent();
  }, []); // Empty dependency array - only run on mount

  return useMemo(() => ({ status, isLoading }), [status, isLoading]);
}
