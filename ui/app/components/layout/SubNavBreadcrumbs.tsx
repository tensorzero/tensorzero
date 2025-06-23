import { Link } from "react-router";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { useBreadcrumbs } from "~/hooks/useBreadcrumbs";

export function SubNavBreadcrumbs() {
  const breadcrumbs = useBreadcrumbs();

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {breadcrumbs.map((bc, idx) => (
          <BreadcrumbItem key={idx}>
            {idx === breadcrumbs.length - 1 ? (
              <BreadcrumbPage>{bc.label}</BreadcrumbPage>
            ) : bc.href ? (
              <>
                <BreadcrumbLink asChild>
                  <Link to={bc.href}>{bc.label}</Link>
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
