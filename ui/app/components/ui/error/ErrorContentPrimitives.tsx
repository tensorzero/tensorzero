import * as React from "react";
import { useAsyncError, isRouteErrorResponse } from "react-router";
import { AlertCircle, type LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/utils/common";

/**
 * Extracts a user-friendly message from an error of any type.
 * Handles React Router's route error responses, standard Errors, and unknown types.
 */
export function getErrorMessage({
  error,
  fallback,
}: {
  /** The error to extract a message from (typically from useAsyncError()) */
  error: unknown;
  /** Default message if error type is unrecognized */
  fallback: string;
}): string {
  if (isRouteErrorResponse(error)) {
    return typeof error.data === "string"
      ? error.data
      : `${error.status} ${error.statusText}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

interface ErrorContentCardProps {
  children: React.ReactNode;
  className?: string;
}

export function ErrorContentCard({
  children,
  className,
}: ErrorContentCardProps) {
  return (
    <Card
      className={cn(
        "w-[26rem] max-w-full rounded-none border-none bg-transparent shadow-none",
        className,
      )}
    >
      {children}
    </Card>
  );
}

interface ErrorContentHeaderProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

export function ErrorContentHeader({
  icon: Icon,
  title,
  description,
}: ErrorContentHeaderProps) {
  return (
    <CardHeader>
      <div className="flex items-center gap-4">
        <Icon className="h-6 w-6 shrink-0 text-red-500 dark:text-red-400" />
        <div className="min-w-0 flex-1">
          <CardTitle className="text-foreground font-medium">{title}</CardTitle>
          <p className="text-muted-foreground mt-1.5 text-sm break-words">
            {description}
          </p>
        </div>
      </div>
    </CardHeader>
  );
}

interface TroubleshootingSectionProps {
  children: React.ReactNode;
}

// Children are auto-numbered as an ordered list (1, 2, 3...)
export function TroubleshootingSection({
  children,
}: TroubleshootingSectionProps) {
  return (
    <CardContent className="border-t p-6">
      <h4 className="text-foreground mb-3 text-sm font-medium">
        What to check:
      </h4>
      <ol className="text-muted-foreground space-y-2 text-sm">
        {React.Children.map(children, (child, index) => (
          <li key={index} className="flex items-start gap-2">
            <span className="bg-muted text-muted-foreground flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs">
              {index + 1}
            </span>
            <span>{child}</span>
          </li>
        ))}
      </ol>
    </CardContent>
  );
}

interface ErrorInlineCodeProps {
  children: React.ReactNode;
}

export function ErrorInlineCode({ children }: ErrorInlineCodeProps) {
  return (
    <code className="bg-muted rounded px-1 py-0.5 font-mono text-xs">
      {children}
    </code>
  );
}

interface StackTraceContentProps {
  stack: string;
}

export function StackTraceContent({ stack }: StackTraceContentProps) {
  return (
    <CardContent className="flex h-40 flex-col border-t p-6">
      <pre className="bg-muted text-muted-foreground min-h-0 flex-1 overflow-auto rounded p-3 font-mono text-xs">
        {stack}
      </pre>
    </CardContent>
  );
}

export function PageErrorContainer({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="pb-page-bottom flex min-h-full items-center justify-center p-8">
      {children}
    </div>
  );
}

export function TableErrorContainer({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="flex justify-center py-16">{children}</div>;
}

interface ErrorNoticeProps {
  icon: LucideIcon;
  title: string;
  description: string;
  /** Use muted gray styling instead of red (e.g., for 404s) */
  muted?: boolean;
}

export function ErrorNotice({
  icon: Icon,
  title,
  description,
  muted = false,
}: ErrorNoticeProps) {
  return (
    <div className="flex w-[26rem] max-w-full flex-col items-center px-8 py-10 text-center">
      <Icon
        className={cn(
          "mb-4 h-10 w-10",
          muted ? "text-muted-foreground" : "text-red-500 dark:text-red-400",
        )}
      />
      <h2 className="text-foreground text-xl font-medium">{title}</h2>
      <p className="text-muted-foreground mt-2 max-w-xs text-sm break-words">
        {description}
      </p>
    </div>
  );
}

/**
 * Convenience wrapper: ErrorNotice inside PageErrorContainer.
 * Use for full-page error states.
 */
export function PageErrorNotice(props: ErrorNoticeProps) {
  return (
    <PageErrorContainer>
      <ErrorNotice {...props} />
    </PageErrorContainer>
  );
}

/**
 * Convenience wrapper: ErrorNotice inside TableErrorContainer.
 * Use for inline table error states (inside TableCell).
 */
export function TableErrorNotice(props: ErrorNoticeProps) {
  return (
    <TableErrorContainer>
      <ErrorNotice {...props} />
    </TableErrorContainer>
  );
}

export function SectionErrorContainer({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="border-border flex justify-center rounded-md border py-12">
      {children}
    </div>
  );
}

/**
 * Convenience wrapper: ErrorNotice inside SectionErrorContainer.
 * Use for section-level error states (BasicInfo, Input, Output, etc.).
 */
export function SectionErrorNotice(props: ErrorNoticeProps) {
  return (
    <SectionErrorContainer>
      <ErrorNotice {...props} />
    </SectionErrorContainer>
  );
}

interface SectionAsyncErrorStateProps {
  defaultMessage?: string;
}

/**
 * Error state for sections using React Router's <Await> component.
 * Must be rendered inside an <Await errorElement={...}> to access the async error.
 * @throws Error if used outside of an <Await errorElement={...}> context
 */
export function SectionAsyncErrorState({
  defaultMessage = "Failed to load data",
}: SectionAsyncErrorStateProps) {
  const error = useAsyncError();

  if (error === undefined) {
    throw new Error(
      "SectionAsyncErrorState must be used inside an <Await errorElement={...}>",
    );
  }

  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading data"
      description={getErrorMessage({ error, fallback: defaultMessage })}
    />
  );
}

interface InlineAsyncErrorProps {
  defaultMessage?: string;
}

/**
 * Inline error for header-level content (BasicInfo, etc.)
 * Uses a subtle inline design that fits within flex layouts without
 * taking over the entire section. Displays icon and error text.
 *
 * Must be rendered inside an <Await errorElement={...}> context.
 */
export function InlineAsyncError({
  defaultMessage = "Failed to load data",
}: InlineAsyncErrorProps) {
  const error = useAsyncError();

  if (error === undefined) {
    throw new Error(
      "InlineAsyncError must be used inside an <Await errorElement={...}>",
    );
  }

  const message = getErrorMessage({ error, fallback: defaultMessage });

  return (
    <div className="inline-flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      <span>{message}</span>
    </div>
  );
}

interface ActionBarAsyncErrorProps {
  defaultMessage?: string;
}

/**
 * Error state for action bars / button groups.
 * Renders as a chip-like element that fits alongside buttons.
 * Uses same height as action bar skeletons (h-8).
 *
 * Must be rendered inside an <Await errorElement={...}> context.
 */
export function ActionBarAsyncError({
  defaultMessage = "Unable to load actions",
}: ActionBarAsyncErrorProps) {
  const error = useAsyncError();

  if (error === undefined) {
    throw new Error(
      "ActionBarAsyncError must be used inside an <Await errorElement={...}>",
    );
  }

  const message = getErrorMessage({ error, fallback: defaultMessage });

  return (
    <div className="inline-flex h-8 w-fit items-center gap-1.5 rounded-md border border-red-200 bg-red-50 px-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      <span>{message}</span>
    </div>
  );
}
