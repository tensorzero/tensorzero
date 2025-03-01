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

interface BreadcrumbConfig {
  // Segments to exclude from breadcrumbs
  excludeSegments: string[];
  // Custom labels for specific segments
  customLabels: { [key: string]: string };
  // Special path handling for specific segments
  specialPaths: {
    [key: string]: (
      segments: string[],
      index: number,
    ) => BreadcrumbSegment | null;
  };
}

// Breadcrumb configuration
const breadcrumbConfig: BreadcrumbConfig = {
  excludeSegments: ["observability", "optimization"],
  customLabels: {
    functions: "Functions",
    inferences: "Inferences",
    episodes: "Episodes",
    "supervised-fine-tuning": "Supervised Fine-tuning",
    datasets: "Datasets",
    datapoints: "Datapoints",
    datapoint: "Datapoint",
  },
  specialPaths: {
    // Handle variants special case
    variants: (segments, index) => {
      if (segments[index + 1]) {
        return {
          label: segments[index + 1],
          href: `/observability/functions/${segments[index - 1]}/variants/${segments[index + 1]}`,
        };
      }
      return null;
    },
    // Handle datapoints special case
    datapoints: (segments, index) => {
      if (segments[index + 1]) {
        return {
          label: segments[index + 1],
          href: `/datasets/${segments[index - 1]}/datapoints/${segments[index + 1]}`,
        };
      }
      return {
        label: "Datapoints",
        href: `/datasets/${segments[index - 1]}/datapoints`,
      };
    },
    // Handle singular datapoint special case
    datapoint: (segments, index) => {
      if (segments[index + 1]) {
        return {
          label: segments[index + 1],
          href: `/datasets/${segments[index - 1]}/datapoint/${segments[index + 1]}`,
        };
      }
      return null;
    },
  },
};

export function SubNavBreadcrumbs() {
  const location = useLocation();
  const pathSegments = location.pathname.split("/").filter(Boolean);

  const getBreadcrumbs = (): BreadcrumbSegment[] => {
    const breadcrumbs: BreadcrumbSegment[] = [];

    // Handle root path
    if (pathSegments.length === 0) {
      return [{ label: "Dashboard" }];
    }

    // Process each path segment
    for (let i = 0; i < pathSegments.length; i++) {
      const segment = pathSegments[i];

      // Skip excluded segments
      if (breadcrumbConfig.excludeSegments.includes(segment)) {
        continue;
      }

      // Check for special path handling
      if (breadcrumbConfig.specialPaths[segment]) {
        const specialSegment = breadcrumbConfig.specialPaths[segment](
          pathSegments,
          i,
        );
        if (specialSegment) {
          breadcrumbs.push(specialSegment);
        }
        continue;
      }

      // Handle IDs and names (segments that follow a known parent)
      if (i > 0) {
        const prevSegment = pathSegments[i - 1];

        if (["functions", "inferences", "episodes"].includes(prevSegment)) {
          breadcrumbs.push({
            label: segment,
            href: `/observability/${prevSegment}/${segment}`,
          });
          continue;
        }

        if (prevSegment === "datasets") {
          breadcrumbs.push({
            label: segment,
            href: `/datasets/${segment}`,
          });
          continue;
        }
      }

      // Add standard breadcrumb if not handled by special cases
      if (breadcrumbConfig.customLabels[segment]) {
        const category =
          segment === "supervised-fine-tuning"
            ? "optimization"
            : segment === "datasets"
              ? ""
              : "observability";
        breadcrumbs.push({
          label: breadcrumbConfig.customLabels[segment],
          href: `/${category}${category ? "/" : ""}${segment}`,
        });
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
