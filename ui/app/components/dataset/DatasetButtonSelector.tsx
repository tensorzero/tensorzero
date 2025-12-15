import { useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button, ButtonIcon } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Table, TablePlus } from "~/components/icons/Icons";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { cn } from "~/utils/common";
import { useDatasetSelector } from "~/hooks/use-dataset-selector";
import { ComboboxMenuItems } from "~/components/ui/combobox";

interface DatasetButtonSelectorProps {
  selected?: string;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  label: string;
  labelClassName?: string;
  inputPlaceholder?: string;
  className?: string;
  allowCreation?: boolean;
  buttonProps?: React.ComponentProps<typeof Button>;
  disabled?: boolean;
}

export function DatasetButtonSelector({
  selected,
  onSelect,
  functionName,
  label,
  labelClassName,
  inputPlaceholder,
  allowCreation = true,
  className,
  buttonProps,
  disabled = false,
}: DatasetButtonSelectorProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const {
    sortedDatasetNames,
    isLoading,
    isError,
    getItemIcon,
    getItemSuffix,
    getItemDataAttributes,
    getSelectedDataset,
  } = useDatasetSelector(functionName);

  const computedPlaceholder = useMemo(() => {
    if (inputPlaceholder) return inputPlaceholder;
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [inputPlaceholder, allowCreation, sortedDatasetNames.length]);

  const selectedDataset = getSelectedDataset(selected);

  const showCreateOption =
    allowCreation &&
    Boolean(inputValue.trim()) &&
    !sortedDatasetNames.some(
      (name) => name.toLowerCase() === inputValue.trim().toLowerCase(),
    );

  const handleSelectItem = (item: string, isNew: boolean) => {
    onSelect(item, isNew);
    setInputValue("");
    setOpen(false);
  };

  return (
    <div className={clsx("flex flex-col space-y-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild disabled={disabled}>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="group w-full justify-between border"
            disabled={disabled}
            {...buttonProps}
          >
            {selected ? (
              <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                {selectedDataset ? (
                  <Table
                    size={16}
                    className="h-4 w-4 shrink-0 text-green-700"
                  />
                ) : (
                  <TablePlus className="h-4 w-4 shrink-0 text-blue-600" />
                )}
                <span className="truncate font-mono text-sm">
                  {selectedDataset?.name ?? selected}
                </span>
                {selectedDataset && (
                  <span className="bg-bg-tertiary text-fg-tertiary ml-auto shrink-0 rounded px-1.5 py-0.5 text-xs">
                    {selectedDataset.count.toLocaleString()}
                  </span>
                )}
              </div>
            ) : (
              <span className="flex flex-row items-center gap-2">
                <ButtonIcon as={Table} variant="tertiary" />
                <span
                  className={cn(
                    "text-fg-secondary flex text-sm",
                    labelClassName,
                  )}
                >
                  {label}
                </span>
              </span>
            )}

            <ButtonIcon
              as={ChevronDown}
              className={clsx(
                "h-4 w-4 shrink-0 transition duration-300 ease-out",
                open ? "-rotate-180" : "rotate-0",
              )}
              variant="tertiary"
            />
          </Button>
        </PopoverTrigger>

        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
          align="start"
        >
          <Command>
            <CommandInput
              placeholder={computedPlaceholder}
              value={inputValue}
              onValueChange={setInputValue}
              className="h-9"
            />

            {isLoading ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                Loading datasets...
              </div>
            ) : isError ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                There was an error loading datasets.
              </div>
            ) : (
              <CommandList>
                <CommandEmpty className="flex items-center justify-center px-4 py-4 text-sm">
                  No datasets found.
                </CommandEmpty>
                <ComboboxMenuItems
                  items={sortedDatasetNames}
                  selected={selected}
                  searchValue={inputValue}
                  onSelectItem={handleSelectItem}
                  showCreateOption={showCreateOption}
                  createHeading="New dataset"
                  existingHeading="Existing"
                  getItemIcon={getItemIcon}
                  getItemSuffix={getItemSuffix}
                  getItemDataAttributes={getItemDataAttributes}
                />
              </CommandList>
            )}
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
