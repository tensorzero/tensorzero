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
  /** Whether a create option is shown above the list */
  hasCreateOption?: boolean;
  /** Called when create option is selected via Enter key (index -1) */
  onSelectCreate?: () => void;
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
  hasCreateOption = false,
  onSelectCreate,
}: UseVirtualizedNavigationOptions): UseVirtualizedNavigationResult {
  // Index -1 represents the create option when hasCreateOption is true
  const minIndex = hasCreateOption ? -1 : 0;
  const [highlightedIndex, setHighlightedIndex] = useState(minIndex);

  // Clamp highlighted index when item count or create option changes
  useEffect(() => {
    setHighlightedIndex((prev) => {
      if (itemCount === 0) return minIndex;
      return Math.max(minIndex, Math.min(prev, itemCount - 1));
    });
  }, [itemCount, minIndex]);

  const resetHighlight = useCallback(() => {
    setHighlightedIndex(minIndex);
  }, [minIndex]);

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
          setHighlightedIndex((prev) =>
            prev > minIndex ? prev - 1 : minIndex,
          );
          break;

        case "Home":
          e.preventDefault();
          setHighlightedIndex(minIndex);
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
          setHighlightedIndex((prev) =>
            Math.max(prev - PAGE_JUMP_SIZE, minIndex),
          );
          break;

        case "Enter":
          e.preventDefault();
          if (highlightedIndex === -1 && hasCreateOption && onSelectCreate) {
            onSelectCreate();
          } else if (highlightedIndex >= 0 && itemCount > 0) {
            onSelect(highlightedIndex);
          }
          break;
      }
    },
    [
      enabled,
      itemCount,
      highlightedIndex,
      onSelect,
      onClose,
      minIndex,
      hasCreateOption,
      onSelectCreate,
    ],
  );

  return {
    highlightedIndex,
    handleKeyDown,
    resetHighlight,
  };
}
