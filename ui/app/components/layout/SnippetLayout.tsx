import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { type ReactNode, useState, useRef, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { clsx } from "clsx";

// Main layout component
interface SnippetLayoutProps {
  children: ReactNode;
}

export function SnippetLayout({ children }: SnippetLayoutProps) {
  return (
    <div className="border-border bg-bg-primary w-full rounded-lg border">
      {children}
    </div>
  );
}

// Content component
interface SnippetContentProps {
  children: ReactNode;
  className?: string;
  maxHeight?: number | "Content";
}

export function SnippetContent({
  children,
  maxHeight = 240,
}: SnippetContentProps) {
  const [expanded, setExpanded] = useState(false);
  const [needsExpansion, setNeedsExpansion] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  // Simple check on initial render and when content changes
  useEffect(() => {
    // Reset expanded state when content changes
    setExpanded(false);

    if (contentRef.current && maxHeight !== "Content") {
      // Simple check if content is taller than maxHeight
      const contentHeight = contentRef.current.scrollHeight;
      setNeedsExpansion(contentHeight > maxHeight);
    } else {
      setNeedsExpansion(false);
    }
  }, [children, maxHeight]);

  return (
    <div className="relative overflow-hidden rounded-b-lg">
      <div
        ref={contentRef}
        style={
          !expanded && needsExpansion && maxHeight !== "Content"
            ? { maxHeight: `${maxHeight}px` }
            : {}
        }
        className={clsx(
          "relative space-y-4",
          !expanded &&
            needsExpansion &&
            maxHeight !== "Content" &&
            "overflow-hidden",
        )}
      >
        {children}
      </div>

      {needsExpansion && !expanded && maxHeight !== "Content" && (
        <div className="from-bg-primary absolute right-0 bottom-0 left-0 flex justify-center bg-gradient-to-t to-transparent pt-8 pb-4">
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
}

export function SnippetMessage({ children }: SnippetMessageProps) {
  return <div className="space-y-2">{children}</div>;
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
  children?: ((activeTab: string) => ReactNode) | ReactNode;
  onTabChange?: (tabId: string) => void;
}

export function SnippetTabs({
  tabs,
  defaultTab,
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
    <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
      <TabsList className="border-border flex w-full justify-start rounded-none border-b p-3">
        {tabs.map((tab) => (
          <TabsTrigger
            key={tab.id}
            value={tab.id}
            className={clsx("flex items-center", tab.indicator && "gap-2")}
          >
            {tab.indicator && tab.indicator !== "none" && (
              <div
                className={clsx(
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
        <TabsContent value={activeTab}>{children(activeTab)}</TabsContent>
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
