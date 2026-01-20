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

interface PageHeaderProps {
  eyebrow?: ReactNode;
  heading?: string;
  name?: string;
  count?: CountValue;
  children?: ReactNode;
  tag?: ReactNode;
}

function CountDisplay({ count }: { count: CountValue }) {
  const resolvedCount = count instanceof Promise ? use(count) : count;
  return (
    <span className="text-fg-muted text-2xl font-medium">
      {resolvedCount.toLocaleString()}
    </span>
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
  type BreadcrumbSegment,
};
