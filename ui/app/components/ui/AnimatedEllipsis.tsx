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

type AnimatedEllipsisProps = {
  /** Animation interval in ms */
  interval?: number;
  /** Reserve width to prevent layout shift */
  reserveWidth?: boolean;
  /** Position absolutely (doesn't affect parent layout) */
  absolute?: boolean;
  className?: string;
};

export function AnimatedEllipsis({
  interval = 400,
  reserveWidth = true,
  absolute = false,
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

  if (absolute) {
    return (
      <span className={cn("absolute left-full", className)}>{dotsContent}</span>
    );
  }

  if (reserveWidth) {
    return (
      <span className={cn("relative inline-block", className)}>
        <ReservedWidth>...</ReservedWidth>
        <span className="absolute left-0">{dotsContent}</span>
      </span>
    );
  }

  return <span className={className}>{dotsContent}</span>;
}
