import { useState, useEffect } from "react";
import { cn } from "~/utils/common";

type ReservedWidthProps = {
  /** The text whose width to reserve (rendered invisibly) */
  children: string;
};

/** Reserves the exact width of the given text without rendering it visibly */
export function ReservedWidth({ children }: ReservedWidthProps) {
  return <span className="invisible">{children}</span>;
}

/** Layout mode for AnimatedEllipsis */
export const EllipsisMode = {
  /** Dynamic width - shifts layout as dots change */
  Dynamic: "dynamic",
  /** Fixed width - reserves space for "..." to prevent layout shift */
  FixedWidth: "fixed-width",
  /** Absolute - positioned outside layout flow */
  Absolute: "absolute",
} as const;

export type EllipsisMode = (typeof EllipsisMode)[keyof typeof EllipsisMode];

type AnimatedEllipsisProps = {
  /** Animation interval in ms */
  interval?: number;
  /** Layout mode */
  mode?: EllipsisMode;
  className?: string;
};

export function AnimatedEllipsis({
  interval = 400,
  mode = EllipsisMode.FixedWidth,
  className,
}: AnimatedEllipsisProps) {
  const [dots, setDots] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setDots((prev) => (prev + 1) % 4);
    }, interval);
    return () => clearInterval(timer);
  }, [interval]);

  const dotsContent = ".".repeat(dots);

  if (mode === EllipsisMode.Absolute) {
    return (
      <span className={cn("absolute left-full", className)}>{dotsContent}</span>
    );
  }

  if (mode === EllipsisMode.FixedWidth) {
    return (
      <span className={cn("relative inline-block", className)}>
        <ReservedWidth>...</ReservedWidth>
        <span className="absolute left-0">{dotsContent}</span>
      </span>
    );
  }

  return <span className={className}>{dotsContent}</span>;
}
