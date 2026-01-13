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

type CountValueType = number | bigint | Promise<number | bigint>;

const CountVariant = {
  Page: "page",
  Section: "section",
} as const;

type CountVariantType = (typeof CountVariant)[keyof typeof CountVariant];

interface PageHeaderProps {
  label?: string;
  heading?: string;
  name?: string;
  count?: CountValueType;
  icon?: ReactNode;
  iconBg?: string;
  children?: ReactNode;
  tag?: ReactNode;
}

// Shared error display for failed count loads - shows "—" with tooltip
function CountErrorTooltip({ variant }: { variant: CountVariantType }) {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load count";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "cursor-help font-medium text-red-500 dark:text-red-400",
            variant === CountVariant.Page ? "text-2xl" : "text-xl",
          )}
        >
          —
        </span>
      </TooltipTrigger>
      <TooltipContent>
        <p>{message}</p>
      </TooltipContent>
    </Tooltip>
  );
}

function PageCountError() {
  return <CountErrorTooltip variant={CountVariant.Page} />;
}

function PageCountValue({ value }: { value: number | bigint }) {
  return (
    <h1 className="text-fg-muted text-2xl font-medium">
      {value.toLocaleString()}
    </h1>
  );
}

function PageCountDisplay({ count }: { count: CountValueType }) {
  if (count instanceof Promise) {
    return (
      <Suspense fallback={<Skeleton className="h-8 w-24" />}>
        <Await resolve={count} errorElement={<PageCountError />}>
          {(resolvedCount) => <PageCountValue value={resolvedCount} />}
        </Await>
      </Suspense>
    );
  }
  return <PageCountValue value={count} />;
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
          {count !== undefined && <PageCountDisplay count={count} />}

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
  count?: CountValueType;
  badge?: {
    name: string;
    tooltip: string;
  };
}

function SectionCountError() {
  return <CountErrorTooltip variant={CountVariant.Section} />;
}

function SectionCountValue({ value }: { value: number | bigint }) {
  return (
    <span className="text-fg-muted text-xl font-medium">
      {value.toLocaleString()}
    </span>
  );
}

function SectionCountDisplay({ count }: { count: CountValueType }) {
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
