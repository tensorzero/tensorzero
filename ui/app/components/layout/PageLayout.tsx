import { Badge } from "~/components/ui/badge";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Suspense, use, type ReactNode } from "react";
import { Link } from "react-router";
import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";
import { ChevronRight } from "lucide-react";

export interface BreadcrumbSegment {
  label: string;
  /** If omitted, segment is rendered as non-clickable text */
  href?: string;
  /** If true, renders label in monospace font (for IDs, names, etc.) */
  isIdentifier?: boolean;
}

/**
 * Renders breadcrumb segments for use in PageHeader eyebrow.
 * Segments with href are clickable links; segments without href are plain text.
 */
function Breadcrumbs({ segments }: { segments: BreadcrumbSegment[] }) {
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

const PageLayout: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div
    className={cn(
      "container mx-auto flex flex-col gap-12 px-8 pt-16 pb-20",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);

type CountValue = number | bigint | Promise<number | bigint>;

interface PageHeaderProps {
  /**
   * Eyebrow content displayed above the title.
   * Can be simple text (for modals) or breadcrumbs (for pages).
   */
  eyebrow?: ReactNode;
  heading?: string;
  name?: string;
  count?: CountValue;
  children?: ReactNode;
  /** Inline badge/tag displayed after the title */
  tag?: ReactNode;
}

function CountDisplay({ count }: { count: CountValue }) {
  const resolvedCount = count instanceof Promise ? use(count) : count;
  return (
    <h1 className="text-fg-muted text-2xl font-medium">
      {resolvedCount.toLocaleString()}
    </h1>
  );
}

const PageHeader: React.FC<PageHeaderProps> = ({
  heading,
  eyebrow,
  name,
  count,
  children,
  tag,
}: PageHeaderProps) => {
  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-3">
        {eyebrow !== undefined && (
          <div className="text-fg-secondary text-sm font-normal">{eyebrow}</div>
        )}
        <div className="flex items-center gap-2">
          {heading !== undefined && (
            <h1 className="text-2xl font-medium">{heading}</h1>
          )}
          {name !== undefined && (
            <h1 className="font-mono text-2xl leading-none font-medium">
              {name}
            </h1>
          )}
          {count !== undefined && (
            <Suspense fallback={<Skeleton className="h-8 w-24" />}>
              <CountDisplay count={count} />
            </Suspense>
          )}

          {tag && <div className="ml-1 flex items-center">{tag}</div>}
        </div>
      </div>

      {/* TODO Use wrapper for this instead - feels strange here */}
      {children && <div className="mt-8 flex flex-col gap-8">{children}</div>}
    </div>
  );
};

const SectionsGroup: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div className={cn("flex flex-col gap-12", className)} {...props}>
    {children}
  </div>
);

const SectionLayout: React.FC<React.ComponentProps<"section">> = ({
  children,
  className,
  ...props
}) => (
  <section className={cn("flex flex-col gap-4", className)} {...props}>
    {children}
  </section>
);

interface SectionHeaderProps extends React.PropsWithChildren {
  heading: string;
  count?: CountValue;
  badge?: {
    name: string;
    tooltip: string;
  };
}

function SectionCountDisplay({ count }: { count: CountValue }) {
  const resolvedCount = count instanceof Promise ? use(count) : count;
  return (
    <span className="text-fg-muted text-xl font-medium">
      {resolvedCount.toLocaleString()}
    </span>
  );
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  heading,
  count,
  badge,
  children,
}) => (
  <h2 className="flex items-center gap-2 text-xl font-medium">
    {heading}

    {count !== undefined && (
      <Suspense fallback={<Skeleton className="h-6 w-12" />}>
        <SectionCountDisplay count={count} />
      </Suspense>
    )}

    {badge && (
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="ml-1 px-2 py-0.5 text-xs font-medium"
          >
            {badge.name}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs">{badge.tooltip}</p>
        </TooltipContent>
      </Tooltip>
    )}

    {children}
  </h2>
);

export {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
  PageLayout,
  Breadcrumbs,
};
