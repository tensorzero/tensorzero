import * as React from "react";
import { cn } from "~/utils/common";
import { ScrollArea } from "~/components/ui/scroll-area";
import { PageSubNav } from "./PageSubNav";

interface LayoutProviderProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function ContentLayout({
  children,
  className,
  ...props
}: LayoutProviderProps) {
  return (
    <div className="flex-1 bg-background-tertiary py-2 pl-0 pr-2">
      <div className={cn("h-[calc(100vh-16px)] w-full", className)} {...props}>
        <ScrollArea className="h-full rounded-md border border-border bg-background-secondary">
          <PageSubNav />
          <div className="px-4">{children}</div>
        </ScrollArea>
      </div>
    </div>
  );
}
