import { useMemo, useCallback } from "react";
import type { Control } from "react-hook-form";
import type { SFTFormValues } from "./types";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import type { ChatCompletionConfig } from "~/types/tensorzero";
import { TemplateDetailsDialog } from "./TemplateDetailsDialog";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { CommandGroup, CommandItem } from "~/components/ui/command";
import { GitBranch } from "lucide-react";
import {
  useCombobox,
  ComboboxInput,
  ComboboxContent,
} from "~/components/ui/combobox";

type VariantSelectorProps = {
  control: Control<SFTFormValues>;
  chatCompletionVariants: Record<string, ChatCompletionConfig>;
};

export function VariantSelector({
  control,
  chatCompletionVariants,
}: VariantSelectorProps) {
  const variantNames = useMemo(
    () => Object.keys(chatCompletionVariants),
    [chatCompletionVariants],
  );
  const hasVariants = variantNames.length > 0;

  return (
    <FormField
      control={control}
      name="variant"
      render={({ field }) => {
        const {
          open,
          searchValue,
          commandRef,
          inputValue,
          closeDropdown,
          handleKeyDown,
          handleInputChange,
          handleBlur,
          handleClick,
        } = useCombobox();

        const filteredVariants = useMemo(() => {
          const query = searchValue.toLowerCase();
          if (!query) return variantNames;
          return variantNames.filter((name) =>
            name.toLowerCase().includes(query),
          );
        }, [searchValue]);

        const handleSelect = useCallback(
          (name: string) => {
            field.onChange(name);
            closeDropdown();
          },
          [field, closeDropdown],
        );

        return (
          <FormItem>
            <FormLabel>Prompt</FormLabel>
            <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
              <Popover open={open}>
                <PopoverAnchor asChild>
                  <ComboboxInput
                    value={inputValue(field.value)}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {}}
                    onClick={handleClick}
                    onBlur={handleBlur}
                    placeholder={
                      hasVariants
                        ? "Select a variant name"
                        : "No variants available"
                    }
                    disabled={!hasVariants}
                    monospace
                    open={open}
                    icon={GitBranch}
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
                    emptyMessage="No variants found."
                  >
                    <CommandGroup>
                      {filteredVariants.map((name) => (
                        <CommandItem
                          key={name}
                          value={name}
                          onSelect={() => handleSelect(name)}
                          className="flex items-center gap-2"
                        >
                          <GitBranch className="h-4 w-4 shrink-0" />
                          <span className="truncate font-mono">{name}</span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </ComboboxContent>
                </PopoverContent>
              </Popover>
              <TemplateDetailsDialog
                variant={field.value}
                disabled={!field.value}
                chatCompletionVariants={chatCompletionVariants}
              />
            </div>
          </FormItem>
        );
      }}
    />
  );
}
