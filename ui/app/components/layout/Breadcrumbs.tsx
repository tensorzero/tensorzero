import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { Link } from "react-router";
import { ChevronRight } from "lucide-react";

export interface BreadcrumbSegment {
  label: string;
  /** If omitted, segment is non-clickable */
  href?: string;
  /** Renders label in monospace font */
  isIdentifier?: boolean;
}

export function Breadcrumbs({ segments }: { segments: BreadcrumbSegment[] }) {
  if (segments.length === 0) return null;

  return (
    <Breadcrumb>
      <BreadcrumbList className="gap-1.5 text-sm">
        {segments.map((segment, index) => (
          <BreadcrumbItem key={segment.href ?? `${index}-${segment.label}`}>
            {segment.href ? (
              <BreadcrumbLink asChild>
                <Link
                  to={segment.href}
                  className={segment.isIdentifier ? "font-mono" : undefined}
                >
                  {segment.label}
                </Link>
              </BreadcrumbLink>
            ) : (
              <span className={segment.isIdentifier ? "font-mono" : undefined}>
                {segment.label}
              </span>
            )}
            {index < segments.length - 1 && (
              <BreadcrumbSeparator>
                <ChevronRight className="h-3 w-3" />
              </BreadcrumbSeparator>
            )}
          </BreadcrumbItem>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
