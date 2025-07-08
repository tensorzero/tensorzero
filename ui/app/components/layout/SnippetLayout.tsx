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
    <div className="bg-bg-primary border-border w-full rounded-lg border">
      {children}
    </div>
  );
}

// Heading component
interface SnippetHeadingProps {
  heading: string;
}

export function SnippetHeading({ heading }: SnippetHeadingProps) {
  return <h3 className="px-5 pt-5 pb-1 text-lg font-medium">{heading}</h3>;
}

// Divider component
export function SnippetDivider() {
  return <div className="border-border h-px w-full border-t" />;
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
    <div className="relative overflow-hidden">
      <div
        ref={contentRef}
        style={
          !expanded && needsExpansion && maxHeight !== "Content"
            ? { maxHeight: `${maxHeight}px` }
            : {}
        }
        className={clsx(
          "py-3",
          !expanded &&
            needsExpansion &&
            maxHeight !== "Content" &&
            "overflow-hidden",
        )}
      >
        {children}
      </div>

      {needsExpansion && !expanded && maxHeight !== "Content" && (
        <div className="from-bg-primary absolute right-0 bottom-0 left-0 flex justify-center rounded-b-lg bg-gradient-to-t to-transparent pt-8 pb-4">
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
  variant?: "default" | "input";
  children?: ReactNode;
  role?: string;
}

export function SnippetMessage({
  variant = "default",
  children,
  role,
}: SnippetMessageProps) {
  // Input variant - contains role and children
  if (variant === "input") {
    return (
      <div className="flex w-full flex-col gap-2 px-5 py-2">
        <div className="text-sm font-medium text-purple-600 capitalize">
          {role}
        </div>
        <div className="border-border flex w-full flex-col gap-4 border-l pl-4">
          {children}
        </div>
      </div>
    );
  }

  // Default variant - simple wrapper with padding
  return (
    <div className="flex w-full flex-col gap-4 overflow-hidden px-5 py-2">
      {children}
    </div>
  );
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
  const defaultTabId = defaultTab || tabs[0]?.id;
  const [activeTab, setActiveTab] = useState(defaultTabId);
  if (!tabs || tabs.length === 0) return null;

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
