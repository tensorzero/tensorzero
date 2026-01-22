import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { ReactNode } from "react";
import { cn } from "~/utils/common";
import {
  Breadcrumbs,
  type BreadcrumbSegment,
} from "~/components/layout/Breadcrumbs";
import {
  PageCount,
  SectionCount,
  type CountValue,
} from "~/components/layout/CountDisplay";

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
        <div className="flex items-center gap-2">
          {title && (
            <h1
              className={cn(
                "text-2xl font-medium",
                !heading && name && "font-mono",
              )}
            >
              {title}
            </h1>
          )}
          {count !== undefined && <PageCount count={count} />}
          {tag && <span className="ml-1 inline-flex items-center">{tag}</span>}
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
      {count !== undefined && <SectionCount count={count} />}
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
