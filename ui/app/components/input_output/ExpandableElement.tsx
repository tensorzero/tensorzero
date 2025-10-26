import { type ReactNode, useState, useRef, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { clsx } from "clsx";
import { ChevronDown, ChevronUp } from "lucide-react";

interface ExpandableElementProps {
  children: ReactNode;
  className?: string;
  maxHeight?: number | "Content";
}

export default function ({
  children,
  maxHeight = 240,
}: ExpandableElementProps) {
  const [expanded, setExpanded] = useState(false);
  const [needsExpansion, setNeedsExpansion] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  // Use an effect to observe size changes
  useEffect(() => {
    // Ensure we don't run this logic if expansion is not possible
    if (maxHeight === "Content") {
      setNeedsExpansion(false);
      return;
    }

    const element = contentRef.current;
    if (!element) return;

    // Define the observer
    const observer = new ResizeObserver(() => {
      // When the size changes, check if the scrollHeight exceeds maxHeight
      const contentHeight = element.scrollHeight;
      setNeedsExpansion(contentHeight > maxHeight);
    });

    // Start observing the element
    observer.observe(element);

    // Cleanup function: stop observing when the component unmounts
    return () => {
      observer.disconnect();
    };
  }, [children, maxHeight]); // Re-run if children or maxHeight prop changes

  return (
    <div className="relative">
      <div
        ref={contentRef}
        style={
          !expanded && needsExpansion && maxHeight !== "Content"
            ? { maxHeight: `${maxHeight}px` }
            : {}
        }
        className={clsx(
          "flex flex-col gap-2",
          !expanded &&
            needsExpansion &&
            maxHeight !== "Content" &&
            "overflow-hidden",
        )}
      >
        {children}
      </div>

      {needsExpansion && !expanded && maxHeight !== "Content" && (
        <div className="from-bg-primary absolute right-0 bottom-0 left-0 flex justify-center bg-gradient-to-t to-transparent pt-8 pb-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpanded(true)}
            className="flex items-center gap-1"
          >
            Show more
            <ChevronDown className="h-4 w-4" />
          </Button>
        </div>
      )}

      {needsExpansion && expanded && maxHeight !== "Content" && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpanded(false)}
            className="mt-4 flex items-center gap-1"
          >
            Show less
            <ChevronUp className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
