import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { type ReactNode } from "react";
import { cn } from "~/utils/common";

// Main layout component
interface SnippetLayoutProps {
  children: ReactNode;
  className?: string;
}

export function SnippetLayout({ children, className }: SnippetLayoutProps) {
  return (
    <div
      className={cn(
        "w-full rounded-lg border border-border bg-bg-primary",
        className,
      )}
    >
      {children}
    </div>
  );
}

// Header components
interface SnippetHeaderProps {
  children?: ReactNode;
  heading?: string;
  description?: string;
  className?: string;
}

export function SnippetHeader({
  children,
  heading,
  description,
  className,
}: SnippetHeaderProps) {
  return (
    <div className={cn("flex flex-col space-y-1.5 px-6 pt-6", className)}>
      {heading && <SnippetHeading>{heading}</SnippetHeading>}
      {description && <SnippetDescription>{description}</SnippetDescription>}
      {children}
    </div>
  );
}

interface SnippetHeadingProps {
  children: ReactNode;
  className?: string;
}

export function SnippetHeading({ children, className }: SnippetHeadingProps) {
  return (
    <h3
      className={cn(
        "text-xl font-semibold leading-none tracking-tight",
        className,
      )}
    >
      {children}
    </h3>
  );
}

interface SnippetDescriptionProps {
  children: ReactNode;
  className?: string;
}

export function SnippetDescription({
  children,
  className,
}: SnippetDescriptionProps) {
  return (
    <div className={cn("mt-1 text-sm text-fg-secondary", className)}>
      {children}
    </div>
  );
}

// Content component
interface SnippetContentProps {
  children: ReactNode;
  className?: string;
}

export function SnippetContent({ children, className }: SnippetContentProps) {
  return <div className={cn("space-y-4 p-6 pt-6", className)}>{children}</div>;
}

// Group component
interface SnippetGroupProps {
  children: ReactNode;
  className?: string;
}

export function SnippetGroup({ children, className }: SnippetGroupProps) {
  return <div className={cn("space-y-4", className)}>{children}</div>;
}

// Message component
interface SnippetMessageProps {
  children: ReactNode;
  className?: string;
}

export function SnippetMessage({ children, className }: SnippetMessageProps) {
  return <div className={cn("space-y-2", className)}>{children}</div>;
}

// Tab components
export interface SnippetTab {
  id: string;
  label: string;
  content?: ReactNode;
}

interface SnippetTabsProps {
  tabs?: SnippetTab[];
  defaultTab?: string;
  className?: string;
  children?: ReactNode;
}

export function SnippetTabs({
  tabs,
  defaultTab,
  className,
  children,
}: SnippetTabsProps) {
  if ((!tabs || tabs.length === 0) && !children) return null;

  if (children) {
    return <div className={cn("w-full", className)}>{children}</div>;
  }

  const defaultTabId = defaultTab || tabs?.[0].id;

  return (
    <Tabs defaultValue={defaultTabId} className={cn("w-full", className)}>
      <TabsList className="mb-2">
        {tabs?.map((tab) => (
          <TabsTrigger key={tab.id} value={tab.id}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {tabs?.map((tab) => (
        <TabsContent key={tab.id} value={tab.id}>
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
}
