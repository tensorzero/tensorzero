import { useLocation } from "react-router";
import { useMemo } from "react";

export function useActivePath() {
  const location = useLocation();
  const pathname = location.pathname;

  // Return a stable object with the function as a property
  // This avoids creating new function references
  return useMemo(
    () => ({
      isActive: (path: string) => {
        if (path === "/") {
          return pathname === "/";
        }
        return pathname.startsWith(path);
      },
    }),
    [pathname],
  );
}
