import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface ErrorContentCardProps {
  children: React.ReactNode;
}

export function ErrorContentCard({ children }: ErrorContentCardProps) {
  return (
    <Card className="max-w-lg border-0 bg-transparent shadow-none">
      {children}
    </Card>
  );
}

interface ErrorContentHeaderProps {
  icon: LucideIcon;
  title: string;
  description: string;
  showBorder?: boolean;
}

export function ErrorContentHeader({
  icon: Icon,
  title,
  description,
  showBorder = true,
}: ErrorContentHeaderProps) {
  return (
    <CardHeader className={showBorder ? "border-b border-neutral-900" : ""}>
      <div className="flex items-center gap-4">
        <Icon className="h-6 w-6 shrink-0 text-red-400" />
        <div className="min-w-0 flex-1">
          <CardTitle className="font-medium text-neutral-100">
            {title}
          </CardTitle>
          <p className="mt-1.5 text-sm text-neutral-400">{description}</p>
        </div>
      </div>
    </CardHeader>
  );
}

interface TroubleshootingSectionProps {
  heading?: string;
  children: React.ReactNode;
}

export function TroubleshootingSection({
  heading = "What to check:",
  children,
}: TroubleshootingSectionProps) {
  return (
    <CardContent className="h-40 p-6">
      <h4 className="mb-3 text-sm font-medium text-neutral-100">{heading}</h4>
      <ol className="space-y-2 text-sm text-neutral-400">
        {React.Children.map(children, (child, index) => (
          <li key={index} className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              {index + 1}
            </span>
            <span>{child}</span>
          </li>
        ))}
      </ol>
    </CardContent>
  );
}

interface InlineCodeProps {
  children: React.ReactNode;
}

export function InlineCode({ children }: InlineCodeProps) {
  return (
    <code className="rounded bg-neutral-800 px-1 py-0.5 font-mono text-xs">
      {children}
    </code>
  );
}

interface StackTraceContentProps {
  stack: string;
}

export function StackTraceContent({ stack }: StackTraceContentProps) {
  return (
    <CardContent className="flex h-40 flex-col p-6">
      <pre className="min-h-0 flex-1 overflow-auto rounded bg-neutral-900 p-3 font-mono text-xs text-neutral-400">
        {stack}
      </pre>
    </CardContent>
  );
}
