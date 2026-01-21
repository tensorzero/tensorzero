import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Suspense, use, type ReactNode } from "react";
import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Breadcrumbs,
  type BreadcrumbSegment,
} from "~/components/layout/Breadcrumbs";

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

function CountDisplay({
  count,
  size = "lg",
  className,
}: {
  count: CountValue;
  size?: "lg" | "md";
  className?: string;
}) {
  const resolvedCount = count instanceof Promise ? use(count) : count;
  return (
    <span
      data-testid="count-display"
      className={cn(
        "text-fg-muted font-medium",
        size === "lg" ? "text-2xl" : "text-xl",
        className,
      )}
    >
      {resolvedCount.toLocaleString()}
    </span>
  );
}

interface PageHeaderProps {
  eyebrow?: ReactNode;
  heading?: string;
  name?: string;
  count?: CountValue;
  tag?: ReactNode;
  children?: ReactNode;
}

function PageHeader({
  eyebrow,
  heading,
  name,
  count,
  tag,
  children,
}: PageHeaderProps) {
  const title = heading ?? name;

  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-3">
        {eyebrow && (
          <div className="text-fg-secondary text-sm font-normal">{eyebrow}</div>
        )}
        <div>
          {title && (
            <h1
              className={cn(
                "inline align-middle text-2xl font-medium",
                !heading && name && "font-mono",
              )}
            >
              {title}
            </h1>
          )}
          {count !== undefined && (
            <Suspense
              fallback={
                <Skeleton className="ml-2 inline-block h-6 w-16 align-middle" />
              }
            >
              <CountDisplay count={count} className="ml-2 align-middle" />
            </Suspense>
          )}
          {tag && (
            <span className="ml-3 inline-flex items-center align-middle">
              {tag}
            </span>
          )}
        </div>
      </div>
      {children && <div className="mt-8 flex flex-col gap-8">{children}</div>}
    </div>
  );
}

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

interface SectionHeaderProps {
  heading: string;
  count?: CountValue;
  badge?: { name: string; tooltip: string };
  children?: ReactNode;
}

function SectionHeader({
  heading,
  count,
  badge,
  children,
}: SectionHeaderProps) {
  return (
    <h2 className="flex items-center gap-2 text-xl font-medium">
      {heading}
      {count !== undefined && (
        <Suspense fallback={<Skeleton className="h-6 w-12" />}>
          <CountDisplay count={count} size="md" />
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
}

export {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
  PageLayout,
  Breadcrumbs,
  type BreadcrumbSegment,
};
