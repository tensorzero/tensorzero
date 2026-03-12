import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "~/utils/common";

interface ScrollFadeContainerProps {
  maxHeight: number;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
}

/**
 * Scrollable container with sticky gradient overlays at the top/bottom edges.
 * Gradients appear only when there is more content to scroll in that direction.
 */
export function ScrollFadeContainer({
  maxHeight,
  children,
  className,
  contentClassName,
}: ScrollFadeContainerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  const updateScrollState = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollUp(el.scrollTop > 1);
    setCanScrollDown(el.scrollTop + el.clientHeight < el.scrollHeight - 1);
  }, []);

  useEffect(() => {
    updateScrollState();
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver(updateScrollState);
    observer.observe(el);
    return () => observer.disconnect();
  }, [updateScrollState]);

  return (
    <div
      ref={scrollRef}
      className={cn("flex flex-1 flex-col overflow-auto rounded-lg", className)}
      style={{ maxHeight }}
      onScroll={updateScrollState}
    >
      <div
        className={cn(
          "from-bg-primary pointer-events-none sticky top-0 z-10 h-4 shrink-0 bg-gradient-to-b to-transparent transition-opacity",
          canScrollUp ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        className={cn(
          "flex flex-col gap-1",
          // Remove inner max-height caps to prevent scrollable-in-scrollable
          "[&_.cm-editor]:!max-h-none [&_pre]:!max-h-none",
          contentClassName,
        )}
      >
        {children}
      </div>
      <div
        className={cn(
          "from-bg-primary pointer-events-none sticky bottom-0 z-10 h-4 shrink-0 bg-gradient-to-t to-transparent transition-opacity",
          canScrollDown ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
