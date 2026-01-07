import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/utils/common";

export const ErrorStyle = {
  Light: "LIGHT",
  Dark: "DARK",
} as const;

export type ErrorStyle = (typeof ErrorStyle)[keyof typeof ErrorStyle];

interface ErrorContentCardProps {
  children: React.ReactNode;
  variant?: ErrorStyle;
  className?: string;
}

/**
 * Container card for error content.
 * - dark: dark background for modal overlay
 * - light: standard card with border for content area
 */
export function ErrorContentCard({
  children,
  variant = ErrorStyle.Light,
  className,
}: ErrorContentCardProps) {
  return (
    <Card
      className={cn(
        "max-w-lg shadow-none",
        variant === ErrorStyle.Dark
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
  showBorder?: boolean;
  variant?: ErrorStyle;
}

export function ErrorContentHeader({
  icon: Icon,
  title,
  description,
  showBorder = true,
  variant = ErrorStyle.Light,
}: ErrorContentHeaderProps) {
  return (
    <CardHeader
      className={cn(
        showBorder &&
          (variant === ErrorStyle.Dark
            ? "border-b border-neutral-900"
            : "border-b"),
      )}
    >
      <div className="flex items-center gap-4">
        <Icon
          className={cn(
            "h-6 w-6 shrink-0",
            variant === ErrorStyle.Dark ? "text-red-400" : "text-red-500",
          )}
        />
        <div className="min-w-0 flex-1">
          <CardTitle
            className={cn(
              "font-medium",
              variant === ErrorStyle.Dark
                ? "text-neutral-100"
                : "text-foreground",
            )}
          >
            {title}
          </CardTitle>
          <p
            className={cn(
              "mt-1.5 text-sm",
              variant === ErrorStyle.Dark
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
  variant?: ErrorStyle;
}

// Children are auto-numbered as an ordered list (1, 2, 3...)
export function TroubleshootingSection({
  children,
  variant = ErrorStyle.Light,
}: TroubleshootingSectionProps) {
  return (
    <CardContent className="h-40 p-6">
      <h4
        className={cn(
          "mb-3 text-sm font-medium",
          variant === ErrorStyle.Dark ? "text-neutral-100" : "text-foreground",
        )}
      >
        What to check:
      </h4>
      <ol
        className={cn(
          "space-y-2 text-sm",
          variant === ErrorStyle.Dark
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {React.Children.map(children, (child, index) => (
          <li key={index} className="flex items-start gap-2">
            <span
              className={cn(
                "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs",
                variant === ErrorStyle.Dark
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
  variant?: ErrorStyle;
}

/**
 * Inline code styling for error messages.
 */
export function ErrorInlineCode({
  children,
  variant = ErrorStyle.Dark,
}: ErrorInlineCodeProps) {
  return (
    <code
      className={cn(
        "rounded px-1 py-0.5 font-mono text-xs",
        variant === ErrorStyle.Dark ? "bg-neutral-800" : "bg-muted",
      )}
    >
      {children}
    </code>
  );
}

interface StackTraceContentProps {
  stack: string;
  variant?: ErrorStyle;
}

/**
 * Scrollable stack trace display.
 */
export function StackTraceContent({
  stack,
  variant = ErrorStyle.Dark,
}: StackTraceContentProps) {
  return (
    <CardContent className="flex h-40 flex-col p-6">
      <pre
        className={cn(
          "min-h-0 flex-1 overflow-auto rounded p-3 font-mono text-xs",
          variant === ErrorStyle.Dark
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
  variant?: ErrorStyle;
}

/**
 * Simple message content for errors without troubleshooting steps.
 */
export function SimpleErrorContent({
  message,
  variant = ErrorStyle.Dark,
}: SimpleErrorContentProps) {
  return (
    <CardContent>
      <p
        className={cn(
          "text-sm",
          variant === ErrorStyle.Dark
            ? "text-neutral-400"
            : "text-muted-foreground",
        )}
      >
        {message}
      </p>
    </CardContent>
  );
}
