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
    <div className="flex-1 bg-bg-tertiary py-2 pl-0 pr-2 max-md:p-0">
      <div className={cn("h-[calc(100vh-16px)] w-full", className)} {...props}>
        <ScrollArea className="h-full rounded-md border border-border bg-bg-secondary max-md:rounded-none max-md:border-none">
          <PageSubNav />
          {children}
        </ScrollArea>
      </div>
    </div>
  );
}
