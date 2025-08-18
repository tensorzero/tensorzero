import * as React from "react";
import { Progress as ProgressPrimitive } from "radix-ui";

import { cn } from "~/utils/common";
import clsx from "clsx";

export function Progress({
  className,
  value,
  updateInterval = 300,
  ease = "linear",
  ...props
}: React.ComponentPropsWithRef<typeof ProgressPrimitive.Root> & {
  updateInterval?: number;
  ease?: "linear" | "ease-in" | "ease-out" | "ease-in-out";
}) {
  return (
    <ProgressPrimitive.Root
      className={cn(
        "bg-primary/20 relative h-2 w-full overflow-hidden rounded-full",
        className,
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator
        style={{
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-expect-error
          "--_scale": `${value || 0}%`,
          "--_duration": `${updateInterval}ms`,
        }}
        className={clsx(
          "bg-primary h-full w-full flex-1 origin-left scale-x-[var(--_scale)] transition-all duration-[var(--_duration)]",
          {
            "ease-in": ease === "ease-in",
            "ease-out": ease === "ease-out",
            "ease-in-out": ease === "ease-in-out",
            "ease-linear": ease === "linear",
          },
        )}
      />
    </ProgressPrimitive.Root>
  );
}
