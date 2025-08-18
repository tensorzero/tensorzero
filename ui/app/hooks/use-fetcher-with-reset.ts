import { useCallback, useEffect, useMemo, useState } from "react";
import { useFetcher } from "react-router";

type FetcherType<T> = ReturnType<typeof useFetcher<T>>;

export type FetcherWithComponentsReset<T> = FetcherType<T> & {
  reset: () => void;
};

/**
 * `useFetcher` does not provide a way to reset its data. This would be useful
 * in cases where the fetcher's state is used to update the UI and some
 * unrelated interaction want to update the UI back to its initial state, e.g.
 * showing messages in a modal that are cleared when the modal is closed so that
 * the previous submission's data is not shown when the modal is opened again.
 */
export function useFetcherWithReset<T = unknown>(
  opts?: Parameters<typeof useFetcher>[0],
): FetcherWithComponentsReset<T> {
  const fetcher = useFetcher<T>(opts);
  const [isReset, setIsReset] = useState(false);

  const reset = useCallback(() => {
    if (fetcher.state === "idle") {
      setIsReset(true);
    }
  }, [fetcher.state]);

  useEffect(() => {
    if (fetcher.state === "idle") {
      setIsReset(false);
    }
  }, [fetcher.state]);

  const data = isReset ? undefined : fetcher.data;

  return useMemo(
    () => ({
      ...fetcher,
      data,
      reset,
    }),
    [fetcher, reset, data],
  );
}
