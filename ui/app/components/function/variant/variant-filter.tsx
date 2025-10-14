import * as React from "react";
import { Check } from "lucide-react";
import { matchSorter } from "match-sorter";
import {
  Combobox,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
  ComboboxProvider,
} from "@ariakit/react";
import { Popover } from "radix-ui";
import { Input } from "~/components/ui/input";
import clsx from "clsx";
import { Separator } from "~/components/ui/separator";

export type VariantData = { color?: string; name: string };

interface VariantFilterProps {
  variants: VariantData[];
  selectedValues: string[];
  setSelectedValues: React.Dispatch<React.SetStateAction<string[]>>;
  disabled?: boolean;
}

export function VariantFilter({
  variants,
  selectedValues,
  setSelectedValues,
  disabled,
}: VariantFilterProps) {
  const [open, setOpen] = React.useState(false);
  const comboboxRef = React.useRef<HTMLInputElement | null>(null);
  const listboxRef = React.useRef<HTMLDivElement | null>(null);
  const [isPending, startTransition] = React.useTransition();
  const [searchValue, setSearchValue] = React.useState("");
  const matches = React.useMemo(
    () =>
      matchSorter(variants, searchValue, {
        keys: ["name"],
      }),
    [searchValue, variants],
  );

  const areAllSelected = selectedValues.length === variants.length;

  return (
    <Popover.Root open={open} onOpenChange={disabled ? undefined : setOpen}>
      <ComboboxProvider
        open={open}
        setOpen={disabled ? undefined : setOpen}
        selectedValue={selectedValues}
        setSelectedValue={
          disabled
            ? undefined
            : (values) => {
                if (values.includes("ALL")) {
                  setSelectedValues((v) =>
                    variants.length === v.length
                      ? []
                      : variants.map((v) => v.name),
                  );
                } else {
                  setSelectedValues(values);
                }
              }
        }
        setValue={
          disabled
            ? undefined
            : (value) => {
                startTransition(() => {
                  setSearchValue(value);
                });
              }
        }
      >
        <ComboboxLabel className="sr-only" render={<label />}>
          Filter by variant
        </ComboboxLabel>
        <Popover.Anchor asChild>
          <Combobox
            ref={comboboxRef}
            placeholder="Filter by variant..."
            className="combobox"
            disabled={disabled}
            render={<Input />}
          />
        </Popover.Anchor>

        <Popover.Content
          asChild
          sideOffset={8}
          aria-busy={isPending}
          side="bottom"
          align="start"
          className="bg-popover text-popover-foreground data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 relative z-50 max-h-96 min-w-[max(8rem,var(--radix-popper-anchor-width))] overflow-auto overscroll-contain rounded-md border p-1 shadow-md"
          onOpenAutoFocus={(event) => event.preventDefault()}
          style={{ zIndex: 1000 }}
          onInteractOutside={(event) => {
            const target = event.target as Element | null;
            const isCombobox = target === comboboxRef.current;
            const inListbox = target && listboxRef.current?.contains(target);
            if (isCombobox || inListbox) {
              event.preventDefault();
            }
          }}
        >
          {matches.length ? (
            <ComboboxList ref={listboxRef} role="listbox" className="listbox">
              <VariantFilterItem
                isSelected={areAllSelected}
                label={areAllSelected ? "Filter all" : "Select all"}
                value="ALL"
              />
              <Separator className="my-1" />
              {matches.map((variant) => {
                const isSelected = selectedValues.includes(variant.name);
                return (
                  <VariantFilterItem
                    isSelected={isSelected}
                    label={
                      <span className="font-mono text-sm">{variant.name}</span>
                    }
                    value={variant.name}
                    key={variant.name}
                    color={variant.color}
                  />
                );
              })}
            </ComboboxList>
          ) : (
            <div className="relative flex select-none gap-2 px-2 py-1.5">
              No results found
            </div>
          )}
        </Popover.Content>
      </ComboboxProvider>
    </Popover.Root>
  );
}

interface VariantFilterItemProps {
  value: string;
  label: React.ReactNode;
  color?: string;
  isSelected: boolean;
  onClick?: (value: string) => void;
}

function VariantFilterItem({
  value,
  label,
  color,
  isSelected,
  onClick,
}: VariantFilterItemProps) {
  return (
    <ComboboxItem
      focusOnHover
      className="data-[active-item]:bg-accent data-[active-item]:text-accent-foreground relative flex h-8 cursor-default select-none items-center justify-between gap-2 rounded-sm px-2 py-1.5 data-[active-item]:outline-none"
      value={value}
      onClick={() => onClick?.(value)}
    >
      <span className="flex items-center gap-2">
        {color && (
          <span
            style={{ "--_bg-color": color } as React.CSSProperties}
            className="rounded-xs block h-2 w-2 bg-[var(--_bg-color)]"
          />
        )}
        <span>{label}</span>
      </span>
      <Check
        className={clsx("h-4 w-4", isSelected ? "opacity-100" : "opacity-0")}
      />
    </ComboboxItem>
  );
}
