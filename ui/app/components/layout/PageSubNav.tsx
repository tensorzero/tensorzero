import { SubNavBreadcrumbs } from "./SubNavBreadcrumbs";

export function PageSubNav() {
  return (
    <div className="supports-[backdrop-filter]:bg-background/60 sticky top-0 z-10 border-b border-border bg-background-secondary px-8 py-2 backdrop-blur">
      <SubNavBreadcrumbs />
    </div>
  );
}
