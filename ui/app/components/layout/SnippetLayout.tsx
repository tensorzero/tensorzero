import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import React, { type ReactNode, useState, useRef, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { clsx } from "clsx";
import { cva } from "class-variance-authority";
import { ChevronDown, ChevronUp } from "lucide-react";

export function SnippetLayout({ children }: React.PropsWithChildren) {
  return (
    <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
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

  // Use an effect to observe size changes
  useEffect(() => {
    // Ensure we don't run this logic if expansion is not possible
    if (maxHeight === "Content") {
      setNeedsExpansion(false);
      return;
    }

    const element = contentRef.current;
    if (!element) return;

    // Define the observer
    const observer = new ResizeObserver(() => {
      // When the size changes, check if the scrollHeight exceeds maxHeight
      const contentHeight = element.scrollHeight;
      setNeedsExpansion(contentHeight > maxHeight);
    });

    // Start observing the element
    observer.observe(element);

    // Cleanup function: stop observing when the component unmounts
    return () => {
      observer.disconnect();
    };
  }, [children, maxHeight]); // Re-run if children or maxHeight prop changes

  return (
    <div className="relative">
      <div
        ref={contentRef}
        style={
          !expanded && needsExpansion && maxHeight !== "Content"
            ? { maxHeight: `${maxHeight}px` }
            : {}
        }
        className={clsx(
          "flex flex-col gap-2",
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
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpanded(true)}
            className="flex items-center gap-1"
          >
            Show more
            <ChevronDown className="h-4 w-4" />
          </Button>
        </div>
      )}

      {needsExpansion && expanded && maxHeight !== "Content" && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpanded(false)}
            className="flex items-center gap-1"
          >
            Show less
            <ChevronUp className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

interface SnippetMessageProps {
  children?: ReactNode;
  role?: "system" | "user" | "assistant";
  action?: ReactNode;
}

const snippetMessageLabel = cva("text-sm font-medium capitalize", {
  variants: {
    role: {
      system: "text-purple-500",
      assistant: "text-emerald-500",
      user: "text-blue-500",
    },
  },
});

const snippetMessage = cva("flex w-full flex-col gap-4 border-l-2 pl-2", {
  variants: {
    role: {
      system: "border-purple-200",
      assistant: "border-emerald-200",
      user: "border-blue-200",
    },
  },
});

export function SnippetMessage({
  children,
  role,
  action,
}: SnippetMessageProps) {
  return (
    <div className="flex w-full flex-col gap-1">
      <div className="relative flex items-center gap-4">
        <div className={snippetMessageLabel({ role })}>{role}</div>
        {action && <div className="absolute left-16">{action}</div>}
      </div>
      <div className={snippetMessage({ role })}>{children}</div>
    </div>
  );
}

export interface SnippetTab {
  id: string;
  label: string | ReactNode;
  content?: ReactNode;
  indicator?: "none" | "empty" | "content" | "fail" | "warning";
}

interface SnippetTabsProps {
  tabs: SnippetTab[];
  defaultTab?: string;
  activeTab?: string;
  children?: ((activeTab: string) => ReactNode) | ReactNode;
  onTabChange?: (tabId: string) => void;
}

const tabIndicator = cva("", {
  variants: {
    indicator: {
      empty: "bg-gray-300",
      content: "bg-green-500",
      fail: "bg-red-500",
      warning: "bg-yellow-500",
    },
  },
});

export function SnippetTabs({
  tabs,
  defaultTab,
  activeTab: controlledActiveTab,
  children,
  onTabChange,
}: SnippetTabsProps) {
  const defaultTabId = defaultTab || tabs[0]?.id;
  const [uncontrolledActiveTab, setUncontrolledActiveTab] =
    useState(defaultTabId);

  // Use controlled tab if provided, otherwise use internal state
  const activeTab = controlledActiveTab ?? uncontrolledActiveTab;

  const handleTabChange = (tabId: string) => {
    if (!controlledActiveTab) {
      setUncontrolledActiveTab(tabId);
    }
    onTabChange?.(tabId);
  };

  return (
    tabs &&
    tabs.length > 0 && (
      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="w-full"
      >
        <TabsList className="flex w-full justify-start rounded-none">
          {tabs.map((tab) => (
            <TabsTrigger
              key={tab.id}
              value={tab.id}
              className={clsx(
                "flex cursor-pointer items-center text-xs",
                tab.indicator && "gap-2",
              )}
            >
              {tab.indicator && tab.indicator !== "none" && (
                <div
                  className={clsx(
                    "h-2 w-2 rounded-full",
                    tabIndicator({
                      indicator: tab.indicator,
                    }),
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
    )
  );
}
