import { Link } from "react-router";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { useBreadcrumbs } from "~/hooks/use-breadcrumbs";

export function SubNavBreadcrumbs() {
  const { segments } = useBreadcrumbs();

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {segments.map((bc, idx) => {
          const className = bc.isIdentifier ? "font-mono" : undefined;

          return (
            <BreadcrumbItem key={idx}>
              {idx === segments.length - 1 ? (
                <BreadcrumbPage className={className}>{bc.label}</BreadcrumbPage>
              ) : bc.href ? (
                <>
                  <BreadcrumbLink asChild>
                    <Link to={bc.href} className={className}>{bc.label}</Link>
                  </BreadcrumbLink>
                  <BreadcrumbSeparator />
                </>
              ) : (
                <>
                  <span className={className}>{bc.label}</span>
                  <BreadcrumbSeparator />
                </>
              )}
            </BreadcrumbItem>
          );
        })}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
