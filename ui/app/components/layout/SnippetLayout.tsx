import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import React, { type ReactNode, useState, useRef, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { clsx } from "clsx";
import { cva } from "class-variance-authority";

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
          <Button variant="outline" size="sm" onClick={() => setExpanded(true)}>
            Show more
          </Button>
        </div>
      )}
    </div>
  );
}

interface SnippetMessageProps {
  children?: ReactNode;
  role?: "system" | "user" | "assistant";
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

export function SnippetMessage({ children, role }: SnippetMessageProps) {
  return (
    <div className="flex w-full flex-col gap-1">
      <div className={snippetMessageLabel({ role })}>{role}</div>
      <div className={snippetMessage({ role })}>{children}</div>
    </div>
  );
}

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
  children,
  onTabChange,
}: SnippetTabsProps) {
  const defaultTabId = defaultTab || tabs[0]?.id;
  const [activeTab, setActiveTab] = useState(defaultTabId);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
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
                "flex items-center text-xs",
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
