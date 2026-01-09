import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Suspense, type ReactNode } from "react";
import { Await, useAsyncError } from "react-router";
import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";

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
  label?: string;
  heading?: string;
  name?: string;
  count?: CountValue;
  icon?: ReactNode;
  iconBg?: string;
  children?: ReactNode;
  tag?: ReactNode;
}

// Error display for failed count load - shows "-" with tooltip
function CountError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load count";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <h1 className="text-fg-muted cursor-help text-2xl font-medium">—</h1>
      </TooltipTrigger>
      <TooltipContent>
        <p className="text-red-600">{message}</p>
      </TooltipContent>
    </Tooltip>
  );
}

function CountValue({ value }: { value: number | bigint }) {
  return (
    <h1 className="text-fg-muted text-2xl font-medium">
      {value.toLocaleString()}
    </h1>
  );
}

function CountDisplay({ count }: { count: CountValue }) {
  if (count instanceof Promise) {
    return (
      <Suspense fallback={<Skeleton className="h-8 w-24" />}>
        <Await resolve={count} errorElement={<CountError />}>
          {(resolvedCount) => <CountValue value={resolvedCount} />}
        </Await>
      </Suspense>
    );
  }
  return <CountValue value={count} />;
}

const PageHeader: React.FC<PageHeaderProps> = ({
  heading,
  label,
  name,
  count,
  icon,
  iconBg = "bg-none",
  children,
  tag,
}: PageHeaderProps) => {
  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-2">
        {label !== undefined && (
          <div className="text-fg-secondary flex items-center gap-1.5 text-sm font-normal">
            {icon && (
              <span
                className={`${iconBg} flex size-5 items-center justify-center rounded-sm`}
              >
                {icon}
              </span>
            )}

            {label}
          </div>
        )}
        <div className="flex items-center gap-2">
          {heading !== undefined && (
            <h1 className="text-2xl font-medium">{heading}</h1>
          )}
          {name !== undefined && (
            <span className="font-mono text-2xl leading-none font-medium">
              {name}
            </span>
          )}
          {count !== undefined && <CountDisplay count={count} />}

          {tag}
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

// Error display for failed section count - shows "-" with tooltip
function SectionCountError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load count";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="text-fg-muted cursor-help text-xl font-medium">—</span>
      </TooltipTrigger>
      <TooltipContent>
        <p className="text-red-600">{message}</p>
      </TooltipContent>
    </Tooltip>
  );
}

function SectionCountValue({ value }: { value: number | bigint }) {
  return (
    <span className="text-fg-muted text-xl font-medium">
      {value.toLocaleString()}
    </span>
  );
}

function SectionCountDisplay({ count }: { count: CountValue }) {
  if (count instanceof Promise) {
    return (
      <Suspense fallback={<Skeleton className="h-6 w-12" />}>
        <Await resolve={count} errorElement={<SectionCountError />}>
          {(resolvedCount) => <SectionCountValue value={resolvedCount} />}
        </Await>
      </Suspense>
    );
  }
  return <SectionCountValue value={count} />;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  heading,
  count,
  badge,
  children,
}) => (
  <h2 className="flex items-center gap-2 text-xl font-medium">
    {heading}

    {count !== undefined && <SectionCountDisplay count={count} />}

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

export { PageHeader, SectionHeader, SectionLayout, SectionsGroup, PageLayout };
