import { useMemo } from "react";
import { useMatches, type RouteHandle, type UIMatch } from "react-router";

export interface BreadcrumbSegment {
  label: string;
  href?: string;
  isIdentifier?: boolean;
}

export function useBreadcrumbs(): {
  segments: BreadcrumbSegment[];
  hideBreadcrumbs: boolean;
} {
  const matches = useMatches() as UIMatch<unknown, RouteHandle | undefined>[];
  return useMemo(
    () => ({
      segments: matches.flatMap(
        (match) =>
          // Each path segment may export a `handle` that appends 0, 1 or more breadcrumbs
          match.handle?.crumb?.(match)?.map((crumb, i, arr) => {
            const label = typeof crumb === "string" ? crumb : crumb.label;
            const isIdentifier = typeof crumb === "string" ? false : crumb.isIdentifier;
            return {
              label,
              isIdentifier,
              // Only show a link for this crumb if it's the last/only breadcrumb for the segment
              // Example: evaluation run adds two crumbs: "Runs" and the run ID. "Runs" would not be a link, since there's no dedicated runs page.
              href: i === arr.length - 1 ? match.pathname : undefined,
            };
          }) ?? [],
      ),
      hideBreadcrumbs: matches.some((match) => match.handle?.hideBreadcrumbs),
    }),
    [matches],
  );
}
