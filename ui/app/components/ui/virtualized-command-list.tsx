import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Command as CommandPrimitive } from "cmdk";
import { cn } from "~/utils/common";

const ITEM_HEIGHT = 36; // px - matches CommandItem py-1.5 + text
const OVERSCAN = 5;

interface VirtualizedCommandListProps {
  children: React.ReactNode;
  className?: string;
  /** Maximum height of the list in pixels. Default: 300 */
  maxHeight?: number;
}

/**
 * A virtualized version of CommandList for rendering large lists efficiently.
 *
 * Usage:
 * ```tsx
 * <VirtualizedCommandList>
 *   <VirtualizedCommandItems
 *     items={items}
 *     renderItem={(item, index) => (
 *       <CommandItem key={item} value={item} onSelect={() => {}}>
 *         {item}
 *       </CommandItem>
 *     )}
 *   />
 * </VirtualizedCommandList>
 * ```
 */
const VirtualizedCommandList = React.forwardRef<
  HTMLDivElement,
  VirtualizedCommandListProps
>(({ className, children, maxHeight = 300 }, ref) => {
  return (
    <CommandPrimitive.List
      ref={ref}
      className={cn("overflow-x-hidden overflow-y-auto", className)}
      style={{ maxHeight }}
    >
      {children}
    </CommandPrimitive.List>
  );
});
VirtualizedCommandList.displayName = "VirtualizedCommandList";

interface VirtualizedCommandItemsProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  /** Height of each item in pixels. Default: 36 */
  itemHeight?: number;
  /** Number of items to render above/below visible area. Default: 5 */
  overscan?: number;
  /** Maximum height of the container. Default: 300 */
  maxHeight?: number;
  className?: string;
}

/**
 * Renders items using virtualization for efficient large list rendering.
 * Must be used inside a Command component with shouldFilter={false}.
 */
function VirtualizedCommandItems<T>({
  items,
  renderItem,
  itemHeight = ITEM_HEIGHT,
  overscan = OVERSCAN,
  maxHeight = 300,
  className,
}: VirtualizedCommandItemsProps<T>) {
  const parentRef = React.useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => itemHeight,
    overscan,
  });

  const virtualItems = virtualizer.getVirtualItems();

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      ref={parentRef}
      className={cn("overflow-y-auto", className)}
      style={{ maxHeight }}
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            transform: `translateY(${virtualItems[0]?.start ?? 0}px)`,
          }}
        >
          {virtualItems.map((virtualRow) => {
            const item = items[virtualRow.index];
            return (
              <div
                key={virtualRow.key}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
              >
                {renderItem(item, virtualRow.index)}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export {
  VirtualizedCommandList,
  VirtualizedCommandItems,
  ITEM_HEIGHT,
  OVERSCAN,
};
