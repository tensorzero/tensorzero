import * as React from "react";
import type { LucideIcon } from "lucide-react";
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
        "max-w-lg shadow-none",
        scope === ErrorScope.App
          ? "border-neutral-800 bg-neutral-950"
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
              "mt-1.5 text-sm",
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
        "h-40 p-6",
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

interface SimpleErrorContentProps {
  message: string;
  scope?: ErrorScope;
}

export function SimpleErrorContent({
  message,
  scope = ErrorScope.Page,
}: SimpleErrorContentProps) {
  return (
    <CardContent
      className={cn(
        scope === ErrorScope.App ? "border-t border-neutral-900" : "border-t",
      )}
    >
      <p
        className={cn(
          "text-sm",
          scope === ErrorScope.App
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {message}
      </p>
    </CardContent>
  );
}
