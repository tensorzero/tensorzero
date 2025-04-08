import * as React from "react";

import { ScrollArea } from "~/components/ui/scroll-area";
import { PageSubNav } from "./PageSubNav";

interface LayoutProviderProps {
  children: React.ReactNode;
}

export function ContentLayout({ children }: LayoutProviderProps) {
  return (
    <div className="bg-bg-tertiary flex-1 py-2 pr-2 pl-0 max-md:p-0">
      <div className="h-[calc(100vh-16px)] w-full">
        <ScrollArea className="border-border bg-bg-secondary h-full rounded-md border max-md:rounded-none max-md:border-none">
          <PageSubNav />
          {children}
        </ScrollArea>
      </div>
    </div>
  );
}
