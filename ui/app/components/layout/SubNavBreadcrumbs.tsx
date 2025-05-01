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
    evaluations: "Evaluations",
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

    for (let i = 0; i < pathSegments.length; i++) {
      const segment = pathSegments[i];

      // 1) Dynamic Evaluations root
      if (segment === "dynamic_evaluations") {
        breadcrumbs.push({
          label: "Dynamic Evaluations",
          href: "/dynamic_evaluations",
        });
        // if that's the only segment, we can return immediately
        if (pathSegments.length === 1) {
          return breadcrumbs;
        }
        continue;
      }

      // 2) Runs list under dynamic_evaluations
      if (
        segment === "runs" &&
        i > 0 &&
        pathSegments[i - 1] === "dynamic_evaluations"
      ) {
        breadcrumbs.push({
          label: "Runs",
          href: "/dynamic_evaluations/runs",
        });
        continue;
      }

      // 3) A specific run ID under /dynamic_evaluations/runs/:id
      if (
        i > 1 &&
        pathSegments[i - 2] === "dynamic_evaluations" &&
        pathSegments[i - 1] === "runs"
      ) {
        breadcrumbs.push({
          label: segment, // the UUID
          href: `/dynamic_evaluations/runs/${segment}`,
        });
        continue;
      }

      // Skip any globally excluded segments
      if (breadcrumbConfig.excludeSegments.includes(segment)) {
        continue;
      }

      // Handle existing specialPaths, IDs, datasets, evaluations, etc...
      if (breadcrumbConfig.specialPaths[segment]) {
        const special = breadcrumbConfig.specialPaths[segment](pathSegments, i);
        if (special) {
          breadcrumbs.push(special);
        }
        continue;
      }

      // ID handling for functions/inferences/episodes
      if (i > 0) {
        const prev = pathSegments[i - 1];

        if (["functions", "inferences", "episodes"].includes(prev)) {
          breadcrumbs.push({
            label: segment,
            href: `/observability/${prev}/${segment}`,
          });
          continue;
        }

        if (prev === "datasets") {
          breadcrumbs.push({
            label: segment === "builder" ? "Builder" : segment,
            href: `/datasets/${segment}`,
          });
          continue;
        }

        if (prev === "evaluations") {
          breadcrumbs.push({
            label: segment,
            href: `/evaluations/${segment}`,
          });
          continue;
        }

        // fallback for datapoint IDs in evaluations
        if (i > 1 && pathSegments[i - 2] === "evaluations") {
          breadcrumbs.push({
            label: segment,
            href: `/evaluations/${pathSegments[i - 1]}/${segment}`,
          });
          continue;
        }
      }

      // Default label/href via customLabels
      if (breadcrumbConfig.customLabels[segment]) {
        const category =
          segment === "supervised-fine-tuning"
            ? "optimization"
            : segment === "datasets" || segment === "evaluations"
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
        {breadcrumbs.map((bc, idx) => (
          <BreadcrumbItem key={idx}>
            {idx === breadcrumbs.length - 1 ? (
              <BreadcrumbPage>{bc.label}</BreadcrumbPage>
            ) : (
              <>
                <BreadcrumbLink asChild>
                  <Link to={bc.href || "#"}>{bc.label}</Link>
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
