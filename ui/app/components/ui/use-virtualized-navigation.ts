import { useCallback, useEffect, useState } from "react";

/** Page jump size for PageUp/PageDown navigation */
const PAGE_JUMP_SIZE = 8;

interface UseVirtualizedNavigationOptions {
  /** Total number of items in the list */
  itemCount: number;
  /** Whether virtualized navigation is enabled */
  enabled: boolean;
  /** Called when an item is selected via Enter key */
  onSelect: (index: number) => void;
  /** Called when Escape is pressed (optional) */
  onClose?: () => void;
}

interface UseVirtualizedNavigationResult {
  /** Currently highlighted item index */
  highlightedIndex: number;
  /** Keyboard event handler for navigation */
  handleKeyDown: (e: React.KeyboardEvent) => void;
  /** Reset highlight to first item */
  resetHighlight: () => void;
}

/**
 * Hook for keyboard navigation in virtualized lists.
 *
 * Handles:
 * - ArrowUp/ArrowDown: Move highlight by one item
 * - Home/End: Jump to first/last item
 * - PageUp/PageDown: Jump by PAGE_JUMP_SIZE items
 * - Enter: Select highlighted item
 * - Escape: Close (if onClose provided)
 */
export function useVirtualizedNavigation({
  itemCount,
  enabled,
  onSelect,
  onClose,
}: UseVirtualizedNavigationOptions): UseVirtualizedNavigationResult {
  const [highlightedIndex, setHighlightedIndex] = useState(0);

  // Clamp highlighted index when item count changes
  useEffect(() => {
    setHighlightedIndex((prev) => {
      if (itemCount === 0) return 0;
      return Math.min(prev, itemCount - 1);
    });
  }, [itemCount]);

  const resetHighlight = useCallback(() => {
    setHighlightedIndex(0);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!enabled) return;

      switch (e.key) {
        case "Escape":
          if (onClose) {
            onClose();
          }
          break;

        case "ArrowDown":
          e.preventDefault();
          setHighlightedIndex((prev) =>
            prev < itemCount - 1 ? prev + 1 : prev,
          );
          break;

        case "ArrowUp":
          e.preventDefault();
          setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : 0));
          break;

        case "Home":
          e.preventDefault();
          setHighlightedIndex(0);
          break;

        case "End":
          e.preventDefault();
          setHighlightedIndex(Math.max(0, itemCount - 1));
          break;

        case "PageDown":
          e.preventDefault();
          setHighlightedIndex((prev) =>
            Math.min(prev + PAGE_JUMP_SIZE, itemCount - 1),
          );
          break;

        case "PageUp":
          e.preventDefault();
          setHighlightedIndex((prev) => Math.max(prev - PAGE_JUMP_SIZE, 0));
          break;

        case "Enter":
          if (itemCount > 0) {
            e.preventDefault();
            onSelect(highlightedIndex);
          }
          break;
      }
    },
    [enabled, itemCount, highlightedIndex, onSelect, onClose],
  );

  return {
    highlightedIndex,
    handleKeyDown,
    resetHighlight,
  };
}
