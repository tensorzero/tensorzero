import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { useBreadcrumbs } from "~/hooks/use-breadcrumbs";
import { Link } from "~/safe-navigation";

export function SubNavBreadcrumbs() {
  const { segments } = useBreadcrumbs();

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {segments.map((bc, idx) => (
          <BreadcrumbItem key={idx}>
            {idx === segments.length - 1 ? (
              <BreadcrumbPage>{bc.label}</BreadcrumbPage>
            ) : bc.href ? (
              <>
                <BreadcrumbLink asChild>
                  <Link unsafeTo={bc.href}>{bc.label}</Link>
                </BreadcrumbLink>
                <BreadcrumbSeparator />
              </>
            ) : (
              <>
                <span>{bc.label}</span>
                <BreadcrumbSeparator />
              </>
            )}
          </BreadcrumbItem>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
