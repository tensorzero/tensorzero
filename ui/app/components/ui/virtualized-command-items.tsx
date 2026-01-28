import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

/**
 * CommandItem height: py-1.5 (12px) + text line height (~24px) = 36px
 * This matches the CommandItem className in command.tsx
 */
const COMMAND_ITEM_HEIGHT = 36;

/** Extra items rendered above/below viewport for smooth scrolling */
const OVERSCAN = 8;

/** Maximum height of the virtualized list */
const MAX_HEIGHT = 300;

interface VirtualizedCommandItemsProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  /** Currently highlighted index for keyboard navigation */
  highlightedIndex?: number;
  className?: string;
}

/**
 * Renders command items using virtualization.
 * Only visible items (plus overscan buffer) exist in the DOM.
 *
 * Requires parent Command to have shouldFilter={false}.
 */
function VirtualizedCommandItems<T>({
  items,
  renderItem,
  highlightedIndex,
  className,
}: VirtualizedCommandItemsProps<T>) {
  const parentRef = React.useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => COMMAND_ITEM_HEIGHT,
    overscan: OVERSCAN,
  });

  React.useEffect(() => {
    if (highlightedIndex !== undefined && highlightedIndex >= 0) {
      virtualizer.scrollToIndex(highlightedIndex, { align: "auto" });
    }
  }, [highlightedIndex, virtualizer]);

  const virtualItems = virtualizer.getVirtualItems();

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      ref={parentRef}
      className={className}
      style={{ maxHeight: MAX_HEIGHT, overflowY: "auto" }}
    >
      <div
        style={{
          height: virtualizer.getTotalSize(),
          position: "relative",
        }}
      >
        {virtualItems.map((virtualRow) => (
          <div
            key={virtualRow.key}
            data-index={virtualRow.index}
            style={{
              position: "absolute",
              top: virtualRow.start,
              width: "100%",
            }}
          >
            {renderItem(items[virtualRow.index], virtualRow.index)}
          </div>
        ))}
      </div>
    </div>
  );
}

export { VirtualizedCommandItems, COMMAND_ITEM_HEIGHT };
