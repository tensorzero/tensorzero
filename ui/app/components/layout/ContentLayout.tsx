import { PageSubNav } from "./PageSubNav";
import { useBreadcrumbs } from "~/hooks/use-breadcrumbs";

export function ContentLayout({ children }: React.PropsWithChildren) {
  const { segments, hideBreadcrumbs } = useBreadcrumbs();
  const pageTitle =
    segments.length > 0
      ? [...segments.map((b) => b.label), "TensorZero"].join(" • ")
      : "Dashboard • TensorZero";

  return (
    <>
      <title>{pageTitle}</title>
      <div className="bg-bg-tertiary w-full min-w-0 flex-1 py-2 pl-0 pr-2 max-md:p-0">
        <div className="h-[calc(100vh-16px)] w-full">
          <div className="border-border bg-bg-secondary h-full overflow-hidden rounded-md border max-md:rounded-none max-md:border-none">
            {!hideBreadcrumbs && <PageSubNav />}
            <div className="h-full overflow-auto">{children}</div>
          </div>
        </div>
      </div>
    </>
  );
}
