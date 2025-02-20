import { SubNavBreadcrumbs } from "./SubNavBreadcrumbs";

export function PageSubNav() {
  return (
    <div className="sticky top-0 z-10 border-b border-border bg-background-secondary px-8 py-2 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <SubNavBreadcrumbs />
    </div>
  );
}
