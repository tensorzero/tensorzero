import { Skeleton } from "~/components/ui/skeleton";
import type { ReactNode } from "react";

export interface StatItem {
  label: string;
  value?: ReactNode;
  detail?: string;
  custom?: ReactNode;
}

const lgColsClass: Record<number, string> = {
  1: "lg:grid-cols-1",
  2: "lg:grid-cols-2",
  3: "lg:grid-cols-3",
  4: "lg:grid-cols-4",
  5: "lg:grid-cols-5",
  6: "lg:grid-cols-6",
};

export function StatsBar({ items }: { items: StatItem[] }) {
  const lgClass = lgColsClass[items.length] ?? "";

  return (
    <div
      className={`bg-bg-secondary border-border grid grid-cols-2 divide-x rounded-lg border ${lgClass}`}
    >
      {items.map((item) => (
        <div key={item.label} className="flex flex-col gap-0.5 px-5 py-3">
          <span className="text-fg-tertiary text-xs">{item.label}</span>
          {item.custom ? (
            item.custom
          ) : (
            <div className="flex items-baseline gap-1.5">
              <span className="text-fg-primary text-sm font-medium">
                {item.value}
              </span>
              {item.detail && (
                <span className="text-fg-muted text-xs">{item.detail}</span>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function StatsBarSkeleton({ count }: { count: number }) {
  const lgClass = lgColsClass[count] ?? "";

  return (
    <div
      className={`bg-bg-secondary border-border grid grid-cols-2 divide-x rounded-lg border ${lgClass}`}
    >
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="flex flex-col gap-1.5 px-5 py-3">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-5 w-12" />
        </div>
      ))}
    </div>
  );
}
