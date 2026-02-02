import { useState, useEffect } from "react";

/**
 * Resolves a promise and returns its value, handling cleanup on unmount or promise change.
 * Returns null while loading or if the promise rejects.
 */
export function useResolvedPromise<T>(promise: Promise<T>): T | null {
  const [value, setValue] = useState<T | null>(null);

  useEffect(() => {
    let cancelled = false;
    setValue(null);
    promise
      .then((result) => {
        if (!cancelled) setValue(result);
      })
      .catch(() => {
        if (!cancelled) setValue(null);
      });
    return () => {
      cancelled = true;
    };
  }, [promise]);

  return value;
}
