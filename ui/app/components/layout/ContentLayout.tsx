import { useMatches, type RouteHandle, type UIMatch } from "react-router";
import { PageSubNav } from "./PageSubNav";
import { useBreadcrumbs } from "~/hooks/use-breadcrumbs";
import { useMemo } from "react";

export function ContentLayout({ children }: React.PropsWithChildren) {
  const { hideBreadcrumbs } = useBreadcrumbs();

  const matches = useMatches() as UIMatch<unknown, RouteHandle | undefined>[];
  const excludeContentWrapper = useMemo(
    () => matches.some((match) => match.handle?.excludeContentWrapper),
    [matches],
  );

  return excludeContentWrapper ? (
    children
  ) : (
    <div className="bg-bg-tertiary w-full min-w-0 flex-1 py-2 pr-2 pl-0 max-md:p-0">
      <div className="h-[calc(100vh-16px)] w-full">
        <div className="border-border bg-bg-secondary h-full overflow-hidden rounded-md border max-md:rounded-none max-md:border-none">
          {!hideBreadcrumbs && <PageSubNav />}
          <div className="h-full overflow-auto">{children}</div>
        </div>
      </div>
    </div>
  );
}
