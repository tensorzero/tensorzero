import { PageSubNav } from "./PageSubNav";
import { useBreadcrumbs } from "~/hooks/useBreadcrumbs";

export function ContentLayout({ children }: React.PropsWithChildren) {
  const breadcrumbs = useBreadcrumbs();
  const pageTitle = [...breadcrumbs.map((b) => b.label), "TensorZero"].join(
    " • ",
  );

  return (
    <>
      <title>{pageTitle}</title>
      <div className="bg-bg-tertiary w-full min-w-0 flex-1 py-2 pr-2 pl-0 max-md:p-0">
        <div className="h-[calc(100vh-16px)] w-full">
          <div className="border-border bg-bg-secondary h-full overflow-hidden rounded-md border max-md:rounded-none max-md:border-none">
            <PageSubNav />
            <div className="h-full overflow-auto">{children}</div>
          </div>
        </div>
      </div>
    </>
  );
}
