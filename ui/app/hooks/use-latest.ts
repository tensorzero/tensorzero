import { useEffect, useRef, type MutableRefObject } from "react";

/**
 * Returns a ref that always contains the latest value.
 * Useful for accessing current values in async callbacks without stale closures.
 *
 * The returned ref is stable (same object across renders), so it doesn't need
 * to be included in useCallback/useEffect dependency arrays.
 */
export function useLatest<T>(value: T): MutableRefObject<T> {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref;
}
