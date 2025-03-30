import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { type ReactNode, useState, useRef, useEffect } from "react";
import { cn } from "~/utils/common";
import { Button } from "~/components/ui/button";

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
    <div className={cn("flex flex-col gap-1 px-6 pb-4 pt-6", className)}>
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
        "text-lg font-medium leading-none tracking-tight",
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
    <div className={cn("text-sm text-fg-secondary", className)}>{children}</div>
  );
}

// Content component
interface SnippetContentProps {
  children: ReactNode;
  className?: string;
  maxHeight?: number;
}

export function SnippetContent({
  children,
  className,
  maxHeight = 360,
}: SnippetContentProps) {
  const [expanded, setExpanded] = useState(false);
  const [needsExpansion, setNeedsExpansion] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  // Simple check on initial render and when content changes
  useEffect(() => {
    // Reset expanded state when content changes
    setExpanded(false);

    if (contentRef.current) {
      // Simple check if content is taller than maxHeight
      const contentHeight = contentRef.current.scrollHeight;
      setNeedsExpansion(contentHeight > maxHeight);
    }
  }, [children, maxHeight]);

  return (
    <div className="relative overflow-hidden rounded-b-lg">
      <div
        ref={contentRef}
        style={
          !expanded && needsExpansion ? { maxHeight: `${maxHeight}px` } : {}
        }
        className={cn(
          "relative space-y-4",
          !expanded && needsExpansion && "overflow-hidden",
          className,
        )}
      >
        {children}
      </div>

      {needsExpansion && !expanded && (
        <div className="absolute bottom-0 left-0 right-0 flex justify-center bg-gradient-to-t from-bg-primary to-transparent pb-4 pt-8">
          <Button variant="outline" size="sm" onClick={() => setExpanded(true)}>
            Show more
          </Button>
        </div>
      )}
    </div>
  );
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
  indicator?: "none" | "empty" | "content" | "fail" | "warning";
}

interface SnippetTabsProps {
  tabs: SnippetTab[];
  defaultTab?: string;
  className?: string;
  children?: ((activeTab: string) => ReactNode) | ReactNode;
  onTabChange?: (tabId: string) => void;
}

export function SnippetTabs({
  tabs,
  defaultTab,
  className,
  children,
  onTabChange,
}: SnippetTabsProps) {
  if (!tabs || tabs.length === 0) return null;

  const defaultTabId = defaultTab || tabs[0].id;
  const [activeTab, setActiveTab] = useState(defaultTabId);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    onTabChange?.(tabId);
  };

  // Function to get indicator color based on indicator type
  const getIndicatorColor = (indicator?: string) => {
    switch (indicator) {
      case "empty":
        return "bg-gray-300";
      case "content":
        return "bg-green-500";
      case "fail":
        return "bg-red-500";
      case "warning":
        return "bg-yellow-500";
      default:
        return "";
    }
  };

  return (
    <Tabs
      value={activeTab}
      onValueChange={handleTabChange}
      className={cn("w-full", className)}
    >
      <TabsList className="flex w-full justify-start rounded-none border-b border-border p-3">
        {tabs.map((tab) => (
          <TabsTrigger
            key={tab.id}
            value={tab.id}
            className={cn("flex items-center", tab.indicator && "gap-2")}
          >
            {tab.indicator && tab.indicator !== "none" && (
              <div
                className={cn(
                  "h-2 w-2 rounded-full",
                  getIndicatorColor(tab.indicator),
                )}
              />
            )}
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {typeof children === "function" ? (
        <TabsContent value={activeTab}>
          {children(activeTab)}
        </TabsContent>
      ) : (
        tabs.map((tab) => (
          <TabsContent key={tab.id} value={tab.id}>
            {tab.content}
          </TabsContent>
        ))
      )}
    </Tabs>
  );
}
