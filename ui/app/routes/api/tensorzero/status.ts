import { useFetcher } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { useEffect, useMemo } from "react";
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

  // biome-ignore lint/correctness/useExhaustiveDependencies: intentionally run only on mount
  useEffect(() => {
    if (statusFetcher.state === "idle" && !statusFetcher.data) {
      statusFetcher.load("/api/tensorzero/status");
    }
  }, []);

  return useMemo(() => ({ status, isLoading }), [status, isLoading]);
}
