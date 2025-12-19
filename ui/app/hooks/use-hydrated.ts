// Implementation taken from remix-utils: https://github.com/sergiodxa/remix-utils

import { useSyncExternalStore } from "react";

function subscribe() {
  return () => {};
}

/**
 * Returns true once React hydration has completed on the client.
 * Use this to disable interactive elements during SSR when event handlers aren't attached.
 *
 * @example
 * const isHydrated = useHydrated();
 * return <button disabled={!isHydrated} onClick={handler}>Click me</button>;
 */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
