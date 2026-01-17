import * as React from "react";
import { type LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/utils/common";

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
    <div className="flex min-h-full items-center justify-center p-8 pb-20">
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
