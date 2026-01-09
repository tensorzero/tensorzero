import * as React from "react";
import { FileQuestion, type LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/utils/common";

/**
 * Error scope based on the nature of the failure:
 * - App: App-level errors (gateway, auth, DB) - dark overlay in ErrorDialog
 * - Page: Page-level errors (404, API failures) - light card, inline or standalone
 */
export const ErrorScope = {
  App: "APP",
  Page: "PAGE",
} as const;

export type ErrorScope = (typeof ErrorScope)[keyof typeof ErrorScope];

interface ErrorContentCardProps {
  children: React.ReactNode;
  scope?: ErrorScope;
  className?: string;
}

/**
 * Container card for error content.
 * Card border is determined by scope: App floats without border, Page has border.
 */
export function ErrorContentCard({
  children,
  scope = ErrorScope.Page,
  className,
}: ErrorContentCardProps) {
  return (
    <Card
      className={cn(
        "w-[26rem] max-w-full shadow-none",
        scope === ErrorScope.App
          ? "rounded-none border-none bg-transparent"
          : "bg-card border",
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
  scope?: ErrorScope;
}

/**
 * Header for error content. Colors adapt to scope.
 * Separator border (if needed) is handled by body components via border-t.
 */
export function ErrorContentHeader({
  icon: Icon,
  title,
  description,
  scope = ErrorScope.Page,
}: ErrorContentHeaderProps) {
  return (
    <CardHeader>
      <div className="flex items-center gap-4">
        <Icon
          className={cn(
            "h-6 w-6 shrink-0",
            scope === ErrorScope.App ? "text-red-400" : "text-red-500",
          )}
        />
        <div className="min-w-0 flex-1">
          <CardTitle
            className={cn(
              "font-medium",
              scope === ErrorScope.App ? "text-neutral-100" : "text-foreground",
            )}
          >
            {title}
          </CardTitle>
          <p
            className={cn(
              "mt-1.5 text-sm break-words",
              scope === ErrorScope.App
                ? "text-neutral-400"
                : "text-muted-foreground",
            )}
          >
            {description}
          </p>
        </div>
      </div>
    </CardHeader>
  );
}

interface TroubleshootingSectionProps {
  children: React.ReactNode;
  scope?: ErrorScope;
}

// Children are auto-numbered as an ordered list (1, 2, 3...)
export function TroubleshootingSection({
  children,
  scope = ErrorScope.Page,
}: TroubleshootingSectionProps) {
  return (
    <CardContent
      className={cn(
        "p-6",
        scope === ErrorScope.App ? "border-t border-neutral-900" : "border-t",
      )}
    >
      <h4
        className={cn(
          "mb-3 text-sm font-medium",
          scope === ErrorScope.App ? "text-neutral-100" : "text-foreground",
        )}
      >
        What to check:
      </h4>
      <ol
        className={cn(
          "space-y-2 text-sm",
          scope === ErrorScope.App
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {React.Children.map(children, (child, index) => (
          <li key={index} className="flex items-start gap-2">
            <span
              className={cn(
                "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs",
                scope === ErrorScope.App
                  ? "bg-neutral-800 text-neutral-300"
                  : "bg-muted text-muted-foreground",
              )}
            >
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
  scope?: ErrorScope;
}

export function ErrorInlineCode({
  children,
  scope = ErrorScope.Page,
}: ErrorInlineCodeProps) {
  return (
    <code
      className={cn(
        "rounded px-1 py-0.5 font-mono text-xs",
        scope === ErrorScope.App ? "bg-neutral-800" : "bg-muted",
      )}
    >
      {children}
    </code>
  );
}

interface StackTraceContentProps {
  stack: string;
  scope?: ErrorScope;
}

export function StackTraceContent({
  stack,
  scope = ErrorScope.Page,
}: StackTraceContentProps) {
  return (
    <CardContent
      className={cn(
        "flex h-40 flex-col p-6",
        scope === ErrorScope.App ? "border-t border-neutral-900" : "border-t",
      )}
    >
      <pre
        className={cn(
          "min-h-0 flex-1 overflow-auto rounded p-3 font-mono text-xs",
          scope === ErrorScope.App
            ? "bg-neutral-900 text-neutral-400"
            : "bg-muted text-muted-foreground",
        )}
      >
        {stack}
      </pre>
    </CardContent>
  );
}

interface PageErrorStackProps {
  icon: LucideIcon;
  title: string;
  description: string;
  scope?: ErrorScope;
  /** Use muted gray styling instead of red (e.g., for 404s) */
  muted?: boolean;
}

/**
 * Centered container for page-level error displays.
 * Used by error boundaries to center content in the available space.
 */
export function PageErrorContainer({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-full items-center justify-center p-8 pb-20">
      {children}
    </div>
  );
}

/**
 * Standard 404 "Page Not Found" display.
 * Reusable across root, layout, and route error boundaries.
 */
export function NotFoundDisplay() {
  return (
    <PageErrorContainer>
      <PageErrorStack
        icon={FileQuestion}
        title="Page Not Found"
        description="The page you're looking for doesn't exist."
        scope={ErrorScope.Page}
        muted
      />
    </PageErrorContainer>
  );
}

/**
 * Standalone error display for simple errors without troubleshooting steps.
 * Vertical layout: large icon above, centered text below.
 * No card border for cleaner appearance.
 */
export function PageErrorStack({
  icon: Icon,
  title,
  description,
  scope = ErrorScope.Page,
  muted = false,
}: PageErrorStackProps) {
  const iconColor = muted
    ? "text-neutral-300"
    : scope === ErrorScope.App
      ? "text-red-400"
      : "text-red-500";

  return (
    <div
      className={cn(
        "flex w-[26rem] max-w-full flex-col items-center px-8 py-10 text-center",
        scope === ErrorScope.App && "bg-transparent",
      )}
    >
      <Icon className={cn("mb-4 h-10 w-10", iconColor)} />
      <h2
        className={cn(
          "text-xl font-medium",
          scope === ErrorScope.App ? "text-neutral-100" : "text-foreground",
        )}
      >
        {title}
      </h2>
      <p
        className={cn(
          "mt-2 max-w-xs text-sm break-words",
          scope === ErrorScope.App
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {description}
      </p>
    </div>
  );
}
