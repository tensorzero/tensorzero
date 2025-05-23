import * as React from "react";
import { PageSubNav } from "./PageSubNav";

interface LayoutProviderProps {
  children: React.ReactNode;
}

export function ContentLayout({ children }: LayoutProviderProps) {
  return (
    <div className="bg-bg-tertiary w-full min-w-0 flex-1 py-2 pl-0 pr-2 max-md:p-0">
      <div className="h-[calc(100vh-16px)] w-full">
        <div className="border-border bg-bg-secondary h-full overflow-hidden rounded-md border max-md:rounded-none max-md:border-none">
          <PageSubNav />
          <div className="h-full overflow-auto">{children}</div>
        </div>
      </div>
    </div>
  );
}
