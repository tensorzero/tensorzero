import { Suspense } from "react";
import { Await, useAsyncError } from "react-router";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";

/**
 * Count value type for page and section headers.
 * Supports both resolved values and promises for streaming data.
 */
export type CountValue = number | bigint | Promise<number | bigint | undefined>;

const CountVariant = {
  Page: "page",
  Section: "section",
} as const;

type CountVariantType = (typeof CountVariant)[keyof typeof CountVariant];

/**
 * Error display for failed count loads - shows "—" with tooltip.
 * Must be used inside an <Await> errorElement.
 */
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

/**
 * Displays a resolved count value with page-level styling.
 */
function PageCountValue({ value }: { value: number | bigint }) {
  return (
    <span
      className="text-fg-muted text-2xl font-medium"
      data-testid="count-display"
    >
      {value.toLocaleString()}
    </span>
  );
}

/**
 * Displays a resolved count value with section-level styling.
 */
function SectionCountValue({ value }: { value: number | bigint }) {
  return (
    <span
      className="text-fg-muted text-xl font-medium"
      data-testid="count-display"
    >
      {value.toLocaleString()}
    </span>
  );
}

/**
 * Renders a page-level count, handling both resolved values and promises.
 * Automatically wraps promises with Suspense/Await and error handling.
 */
export function PageCount({ count }: { count: CountValue }) {
  if (count instanceof Promise) {
    return (
      <Suspense fallback={<Skeleton className="h-6 w-16" />}>
        <Await
          resolve={count}
          errorElement={<CountErrorTooltip variant={CountVariant.Page} />}
        >
          {(resolvedCount) =>
            resolvedCount != null ? (
              <PageCountValue value={resolvedCount} />
            ) : null
          }
        </Await>
      </Suspense>
    );
  }
  return <PageCountValue value={count} />;
}

/**
 * Renders a section-level count, handling both resolved values and promises.
 * Automatically wraps promises with Suspense/Await and error handling.
 */
export function SectionCount({ count }: { count: CountValue }) {
  if (count instanceof Promise) {
    return (
      <Suspense fallback={<Skeleton className="h-5 w-12" />}>
        <Await
          resolve={count}
          errorElement={<CountErrorTooltip variant={CountVariant.Section} />}
        >
          {(resolvedCount) =>
            resolvedCount != null ? (
              <SectionCountValue value={resolvedCount} />
            ) : null
          }
        </Await>
      </Suspense>
    );
  }
  return <SectionCountValue value={count} />;
}
