import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/utils/common";

/**
 * Error context based on the nature of the failure:
 * - App: App-level errors (gateway, auth, DB) - dark overlay in ErrorDialog
 * - Page: Page-level errors (404, API failures) - light card, inline or standalone
 */
export const ErrorContext = {
  App: "APP",
  Page: "PAGE",
} as const;

export type ErrorContext = (typeof ErrorContext)[keyof typeof ErrorContext];

interface ErrorContentCardProps {
  children: React.ReactNode;
  context?: ErrorContext;
  className?: string;
}

/**
 * Container card for error content.
 * Card border is determined by context: App floats without border, Page has border.
 */
export function ErrorContentCard({
  children,
  context = ErrorContext.Page,
  className,
}: ErrorContentCardProps) {
  return (
    <Card
      className={cn(
        "max-w-lg shadow-none",
        context === ErrorContext.App
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
  context?: ErrorContext;
}

/**
 * Header for error content. Colors adapt to context.
 * Separator border (if needed) is handled by body components via border-t.
 */
export function ErrorContentHeader({
  icon: Icon,
  title,
  description,
  context = ErrorContext.Page,
}: ErrorContentHeaderProps) {
  return (
    <CardHeader>
      <div className="flex items-center gap-4">
        <Icon
          className={cn(
            "h-6 w-6 shrink-0",
            context === ErrorContext.App ? "text-red-400" : "text-red-500",
          )}
        />
        <div className="min-w-0 flex-1">
          <CardTitle
            className={cn(
              "font-medium",
              context === ErrorContext.App
                ? "text-neutral-100"
                : "text-foreground",
            )}
          >
            {title}
          </CardTitle>
          <p
            className={cn(
              "mt-1.5 text-sm",
              context === ErrorContext.App
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
  context?: ErrorContext;
}

// Children are auto-numbered as an ordered list (1, 2, 3...)
export function TroubleshootingSection({
  children,
  context = ErrorContext.Page,
}: TroubleshootingSectionProps) {
  return (
    <CardContent
      className={cn(
        "h-40 p-6",
        context === ErrorContext.App
          ? "border-t border-neutral-900"
          : "border-t",
      )}
    >
      <h4
        className={cn(
          "mb-3 text-sm font-medium",
          context === ErrorContext.App ? "text-neutral-100" : "text-foreground",
        )}
      >
        What to check:
      </h4>
      <ol
        className={cn(
          "space-y-2 text-sm",
          context === ErrorContext.App
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {React.Children.map(children, (child, index) => (
          <li key={index} className="flex items-start gap-2">
            <span
              className={cn(
                "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs",
                context === ErrorContext.App
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
  context?: ErrorContext;
}

export function ErrorInlineCode({
  children,
  context = ErrorContext.Page,
}: ErrorInlineCodeProps) {
  return (
    <code
      className={cn(
        "rounded px-1 py-0.5 font-mono text-xs",
        context === ErrorContext.App ? "bg-neutral-800" : "bg-muted",
      )}
    >
      {children}
    </code>
  );
}

interface StackTraceContentProps {
  stack: string;
  context?: ErrorContext;
}

export function StackTraceContent({
  stack,
  context = ErrorContext.Page,
}: StackTraceContentProps) {
  return (
    <CardContent
      className={cn(
        "flex h-40 flex-col p-6",
        context === ErrorContext.App
          ? "border-t border-neutral-900"
          : "border-t",
      )}
    >
      <pre
        className={cn(
          "min-h-0 flex-1 overflow-auto rounded p-3 font-mono text-xs",
          context === ErrorContext.App
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
  context?: ErrorContext;
}

export function SimpleErrorContent({
  message,
  context = ErrorContext.Page,
}: SimpleErrorContentProps) {
  return (
    <CardContent
      className={cn(
        context === ErrorContext.App
          ? "border-t border-neutral-900"
          : "border-t",
      )}
    >
      <p
        className={cn(
          "text-sm",
          context === ErrorContext.App
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {message}
      </p>
    </CardContent>
  );
}
