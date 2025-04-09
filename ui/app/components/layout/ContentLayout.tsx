import * as React from "react";
import { cn } from "~/utils/common";
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
    <div className="bg-bg-tertiary w-full min-w-0 flex-1 py-2 pr-2 pl-0 max-md:p-0">
      <div className={cn("h-[calc(100vh-16px)] w-full", className)} {...props}>
        <div className="border-border bg-bg-secondary h-full overflow-hidden rounded-md border max-md:rounded-none max-md:border-none">
          <PageSubNav />
          <div className="h-full overflow-auto">{children}</div>
        </div>
      </div>
    </div>
  );
}
