import { Link, useLocation } from "react-router";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";

interface BreadcrumbSegment {
  label: string;
  href?: string;
}

export function SubNavBreadcrumbs() {
  const location = useLocation();
  const pathSegments = location.pathname.split("/").filter(Boolean);

  const getBreadcrumbs = (): BreadcrumbSegment[] => {
    const breadcrumbs: BreadcrumbSegment[] = [];

    // Handle root path
    if (pathSegments.length === 0) {
      return [{ label: "Dashboard" }];
    }

    // Build breadcrumbs based on path segments
    for (let i = 0; i < pathSegments.length; i++) {
      const segment = pathSegments[i];

      // Skip category segments
      if (segment === "observability" || segment === "optimization") {
        continue;
      }

      // Special cases for different routes
      switch (segment) {
        case "functions":
          breadcrumbs.push({
            label: "Functions",
            href: "/observability/functions",
          });
          break;

        case "inferences":
          breadcrumbs.push({
            label: "Inferences",
            href: "/observability/inferences",
          });
          break;

        case "episodes":
          breadcrumbs.push({
            label: "Episodes",
            href: "/observability/episodes",
          });
          break;

        case "supervised-fine-tuning":
          breadcrumbs.push({
            label: "Supervised Fine-tuning",
            href: "/optimization/supervised-fine-tuning",
          });
          break;

        case "variants":
          if (pathSegments[i + 1]) {
            breadcrumbs.push({
              label: `${pathSegments[i + 1]}`,
              href: `/observability/functions/${pathSegments[i - 1]}/variants/${pathSegments[i + 1]}`,
            });
          }
          break;

        default:
          // Handle IDs and names
          if (i > 0) {
            const prevSegment = pathSegments[i - 1];
            switch (prevSegment) {
              case "inferences":
                breadcrumbs.push({
                  label: `${segment}`,
                  href: `/observability/inferences/${segment}`,
                });
                break;
              case "episodes":
                breadcrumbs.push({
                  label: `${segment}`,
                  href: `/observability/episodes/${segment}`,
                });
                break;
              case "functions":
                breadcrumbs.push({
                  label: `${segment}`,
                  href: `/observability/functions/${segment}`,
                });
                break;
            }
          }
      }
    }

    return breadcrumbs;
  };

  const breadcrumbs = getBreadcrumbs();

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {breadcrumbs.map((breadcrumb, index) => (
          <BreadcrumbItem key={index}>
            {index === breadcrumbs.length - 1 ? (
              <BreadcrumbPage>{breadcrumb.label}</BreadcrumbPage>
            ) : (
              <>
                <BreadcrumbLink asChild>
                  <Link to={breadcrumb.href || "#"}>{breadcrumb.label}</Link>
                </BreadcrumbLink>
                <BreadcrumbSeparator />
              </>
            )}
          </BreadcrumbItem>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
