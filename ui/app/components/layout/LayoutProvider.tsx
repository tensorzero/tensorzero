import * as React from "react";
import { cn } from "~/utils/common";
import { ScrollArea } from "~/components/ui/scroll-area";

interface LayoutProviderProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function LayoutProvider({ children, className, ...props }: LayoutProviderProps) {
  return (
    <div className="flex-1 pr-2 pl-0 py-2 bg-neutral-100">
      <div
        className={cn(
          "h-[calc(100vh-16px)] w-full",
          className
        )}
        {...props}
      >
        <ScrollArea className="h-full border border-neutral-200 rounded-md bg-neutral-50">
          <div className="px-4">
            {children}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
