import { useFetcher } from "react-router";
import { tensorZeroClient } from "~/utils/tensorzero.server";
import { useEffect } from "react";

export async function loader() {
  try {
    const status = await tensorZeroClient.status();
    return status;
  } catch (error) {
    console.error("Failed to fetch TensorZero status:", error);
    return null;
  }
}

/**
 * A hook that fetches the status of the TensorZero Gateway.
 */

export function useTensorZeroStatusFetcher() {
  const statusFetcher = useFetcher();
  const status = statusFetcher.data;

  useEffect(() => {
    if (statusFetcher.state === "idle" && !statusFetcher.data) {
      statusFetcher.load("/api/tensorzero/status");
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { status, isLoading: statusFetcher.state === "loading" };
}
