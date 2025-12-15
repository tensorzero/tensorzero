import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { useCallback, useMemo } from "react";
import { ComboboxInput } from "./ComboboxInput";
import { ComboboxContent } from "./ComboboxContent";
import { ComboboxHint } from "./ComboboxHint";
import { ComboboxMenuItems } from "./ComboboxMenuItems";
import { useCombobox } from "./use-combobox";

type ComboboxProps = {
  selected: string | null;
  onSelect: (value: string, isNew: boolean) => void;
  items: string[];
  getItemIcon?: (item: string | null, isSelected: boolean) => React.ReactNode;
  getItemSuffix?: (item: string | null) => React.ReactNode;
  getItemDataAttributes?: (item: string) => Record<string, string>;
  placeholder: string;
  emptyMessage: string;
  disabled?: boolean;
  name?: string;
  ariaLabel?: string;
  allowCreation?: boolean;
  creationHint?: string;
  createHeading?: string;
  loading?: boolean;
  loadingMessage?: string;
  error?: boolean;
  errorMessage?: string;
};

export function Combobox({
  selected,
  onSelect,
  items,
  getItemIcon,
  getItemSuffix,
  getItemDataAttributes,
  placeholder,
  emptyMessage,
  disabled = false,
  name,
  ariaLabel,
  allowCreation = false,
  creationHint,
  createHeading = "Create new",
  loading = false,
  loadingMessage = "Loading...",
  error = false,
  errorMessage = "An error occurred.",
}: ComboboxProps) {
  const {
    open,
    searchValue,
    commandRef,
    getInputValue,
    closeDropdown,
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  const filteredItems = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return items;
    return items.filter((item) => item.toLowerCase().includes(query));
  }, [items, searchValue]);

  const handleSelectItem = useCallback(
    (item: string, isNew: boolean) => {
      onSelect(item, isNew);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  const showCreateOption =
    allowCreation &&
    Boolean(searchValue.trim()) &&
    !items.some(
      (item) => item.toLowerCase() === searchValue.trim().toLowerCase(),
    );

  const inputPrefix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    const isSelected = Boolean(selected && !searchValue);
    return getItemIcon?.(item, isSelected);
  }, [selected, searchValue, getItemIcon]);

  const inputSuffix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    return getItemSuffix?.(item);
  }, [selected, searchValue, getItemSuffix]);

  return (
    <div className="w-full">
      {name && <input type="hidden" name={name} value={selected ?? ""} />}
      <Popover open={open}>
        <PopoverAnchor asChild>
          <ComboboxInput
            value={getInputValue(selected)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onClick={handleClick}
            onBlur={handleBlur}
            placeholder={placeholder}
            disabled={disabled}
            open={open}
            prefix={inputPrefix}
            suffix={inputSuffix}
            ariaLabel={ariaLabel}
          />
        </PopoverAnchor>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] p-0"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
        >
          <ComboboxContent
            ref={commandRef}
            emptyMessage={emptyMessage}
            showEmpty={!showCreateOption && !loading && !error}
          >
            {loading ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                {loadingMessage}
              </div>
            ) : error ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                {errorMessage}
              </div>
            ) : (
              <ComboboxMenuItems
                items={filteredItems}
                selected={selected}
                searchValue={searchValue}
                onSelectItem={handleSelectItem}
                showCreateOption={showCreateOption}
                createHeading={createHeading}
                existingHeading="Existing"
                getItemIcon={getItemIcon}
                getItemSuffix={getItemSuffix}
                getItemDataAttributes={getItemDataAttributes}
              />
            )}
          </ComboboxContent>
          {creationHint && !showCreateOption && !loading && !error && (
            <ComboboxHint>{creationHint}</ComboboxHint>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
