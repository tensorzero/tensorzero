import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button, ButtonIcon } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandList,
} from "~/components/ui/command";
import { ComboboxMenuItems } from "~/components/ui/combobox/ComboboxMenuItems";
import { useVirtualizedNavigation } from "~/components/ui/use-virtualized-navigation";
import clsx from "clsx";

/** Default threshold for enabling virtualization */
const DEFAULT_VIRTUALIZE_THRESHOLD = 100;

export interface ButtonSelectRenderTriggerProps {
  open: boolean;
}

export interface ButtonSelectProps {
  /** Items to display in the dropdown */
  items: string[];
  /** Called when an item is selected */
  onSelect: (item: string, isNew: boolean) => void;
  /** Currently selected item (for highlighting in list) */
  selected?: string | null;
  /** Content to render inside the trigger button, or a render function receiving { open } */
  trigger:
    | React.ReactNode
    | ((props: ButtonSelectRenderTriggerProps) => React.ReactNode);
  /** Message shown when no items match the search */
  emptyMessage: string;
  /** Whether the select is disabled */
  disabled?: boolean;
  /** Additional className for the trigger button */
  triggerClassName?: string;
  /** Whether the select is in a loading state */
  isLoading?: boolean;
  /** Message shown while loading */
  loadingMessage?: string;
  /** Whether there was an error loading items */
  isError?: boolean;
  /** Message shown on error */
  errorMessage?: string;
  /** Heading for the create option group */
  createHeading?: string;
  /** Heading for the existing items group (shown when create option visible) */
  existingHeading?: string;
  /** Render prefix content for each item */
  getPrefix?: (item: string | null, isSelected: boolean) => React.ReactNode;
  /** Render suffix content for each item */
  getSuffix?: (item: string | null) => React.ReactNode;
  /** Get data attributes for each item */
  getItemDataAttributes?: (item: string) => Record<string, string>;
  /** Whether to show the search input (default: true) */
  searchable?: boolean;
  /** Placeholder text for the search input (required when searchable is true) */
  placeholder?: string;
  /** Whether to allow creating new items (only works when searchable is true) */
  creatable?: boolean;
  /** Alignment of the popover relative to trigger (default: "start") */
  align?: "start" | "center" | "end";
  /** Additional className for the popover menu */
  menuClassName?: string;
  /**
   * Number of items at which virtualization is enabled.
   * Set to 0 to always virtualize, or Infinity to never virtualize.
   * Default: 100
   */
  virtualizeThreshold?: number;
}

export function ButtonSelect({
  items,
  onSelect,
  selected,
  trigger,
  emptyMessage,
  disabled = false,
  triggerClassName,
  isLoading = false,
  loadingMessage = "Loading...",
  isError = false,
  errorMessage = "An error occurred.",
  createHeading = "Create new",
  existingHeading = "Existing",
  getPrefix,
  getSuffix,
  getItemDataAttributes,
  searchable = true,
  placeholder,
  creatable = false,
  align = "start",
  menuClassName,
  virtualizeThreshold = DEFAULT_VIRTUALIZE_THRESHOLD,
}: ButtonSelectProps) {
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");

  const filteredItems = useMemo(() => {
    if (!searchable || !searchValue.trim()) {
      return items;
    }
    const search = searchValue.toLowerCase().trim();
    return items.filter((item) => item.toLowerCase().includes(search));
  }, [items, searchValue, searchable]);

  const shouldVirtualize = filteredItems.length >= virtualizeThreshold;

  const showCreateOption =
    searchable &&
    creatable &&
    Boolean(searchValue.trim()) &&
    !items.some(
      (item) => item.toLowerCase() === searchValue.trim().toLowerCase(),
    );

  const showMenu = !isLoading && !isError;

  const handleSelectItem = useCallback(
    (item: string, isNew: boolean) => {
      onSelect(item, isNew);
      setSearchValue("");
      setOpen(false);
    },
    [onSelect],
  );

  // Virtualized navigation hook
  const {
    highlightedIndex,
    handleKeyDown: virtualizedHandleKeyDown,
    resetHighlight,
  } = useVirtualizedNavigation({
    itemCount: filteredItems.length,
    enabled: shouldVirtualize,
    onSelect: (index) => {
      const item = filteredItems[index];
      if (item) {
        handleSelectItem(item, false);
      }
    },
    hasCreateOption: showCreateOption,
    onSelectCreate: () => {
      handleSelectItem(searchValue.trim(), true);
    },
  });

  // Reset highlight when dropdown opens or search changes
  useEffect(() => {
    if (open) {
      resetHighlight();
    }
  }, [open, searchValue, resetHighlight]);

  const triggerContent =
    typeof trigger === "function" ? trigger({ open }) : trigger;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild disabled={disabled}>
        <Button
          variant="outline"
          size="sm"
          role="combobox"
          aria-expanded={open}
          className={triggerClassName}
          disabled={disabled}
        >
          {triggerContent}
          <ButtonIcon
            as={ChevronDown}
            className={clsx("h-4 w-4 shrink-0", open && "-rotate-180")}
            variant="tertiary"
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        className={clsx(
          "w-[var(--radix-popover-trigger-width)] min-w-64 p-0",
          menuClassName,
        )}
        align={align}
      >
        <Command
          shouldFilter={false}
          onKeyDown={shouldVirtualize ? virtualizedHandleKeyDown : undefined}
        >
          {searchable && (
            <CommandInput
              placeholder={placeholder}
              value={searchValue}
              onValueChange={setSearchValue}
              className="h-9"
            />
          )}

          {isLoading && (
            <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
              {loadingMessage}
            </div>
          )}

          {isError && (
            <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
              {errorMessage}
            </div>
          )}

          {showMenu && (
            <CommandList>
              <CommandEmpty>{emptyMessage}</CommandEmpty>
              <ComboboxMenuItems
                items={filteredItems}
                selectedValue={selected}
                searchValue={searchValue}
                onSelectItem={handleSelectItem}
                showCreateOption={showCreateOption}
                createHeading={createHeading}
                existingHeading={existingHeading}
                getPrefix={getPrefix}
                getSuffix={getSuffix}
                getItemDataAttributes={getItemDataAttributes}
                virtualize={shouldVirtualize}
                highlightedIndex={highlightedIndex}
              />
            </CommandList>
          )}
        </Command>
      </PopoverContent>
    </Popover>
  );
}
