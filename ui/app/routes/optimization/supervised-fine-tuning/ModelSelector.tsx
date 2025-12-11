import type { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import type { SFTFormValues } from "./types";
import { ModelOptionSchema, type ModelOption } from "./model_options";
import { useState, useRef, useMemo, useEffect, useCallback } from "react";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { Check, ChevronDown, Info } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "~/components/ui/command";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Input } from "~/components/ui/input";
import clsx from "clsx";
import { ModelBadge } from "~/components/model/ModelBadge";
import { formatProvider } from "~/utils/providers";

const providers = ModelOptionSchema.shape.provider.options;

const RADIX_POPPER_SELECTOR = "[data-radix-popper-content-wrapper]";
const RADIX_SELECT_SELECTOR = "[data-radix-select-content]";
const COMMAND_MENU_KEYS = ["ArrowDown", "ArrowUp", "Enter"];

const INPUT_RIGHT_CONTENT_PADDING_PX = 24;

function createCustomModel(
  name: string,
  provider: ModelOption["provider"],
): ModelOption {
  return { displayName: name, name, provider };
}

function filterModelsByName(
  models: ModelOption[],
  query: string,
): ModelOption[] {
  if (!query) return models;
  const lowerQuery = query.toLowerCase();
  return models.filter((m) => m.displayName.toLowerCase().includes(lowerQuery));
}

function isCustomModel(
  value: ModelOption | undefined,
  predefinedModels: ModelOption[],
): boolean {
  if (!value) return false;
  return !predefinedModels.some((m) => m.name === value.name);
}

function isModelSelected(
  fieldValue: ModelOption | undefined,
  model: ModelOption,
): boolean {
  return (
    fieldValue?.name === model.name && fieldValue?.provider === model.provider
  );
}

function isFocusWithinPopover(relatedTarget: Element | null): boolean {
  return Boolean(
    relatedTarget?.closest(RADIX_POPPER_SELECTOR) ||
      relatedTarget?.closest(RADIX_SELECT_SELECTOR),
  );
}

type ModelProviderSelectProps = {
  value: ModelOption["provider"];
  onChange: (value: ModelOption["provider"]) => void;
  onOpenChange: (open: boolean) => void;
};

function ModelProviderSelect({
  value,
  onChange,
  onOpenChange,
}: ModelProviderSelectProps) {
  return (
    <Select value={value} onOpenChange={onOpenChange} onValueChange={onChange}>
      <SelectTrigger
        className="border-border data-[state=open]:border-border-accent h-6 w-auto gap-1 pr-1 pl-2 text-xs focus:ring-0"
        onClick={(e) => e.stopPropagation()}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {providers.map((p) => (
          <SelectItem key={p} value={p}>
            {formatProvider(p).name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

type ModelSelectorProps = {
  control: Control<SFTFormValues>;
  models: ModelOption[];
};

export function ModelSelector({
  control,
  models: predefinedModels,
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [providerSelectOpen, setProviderSelectOpen] = useState(false);
  const [searchValue, setSearchValue] = useState<string | null>(null);
  const [customProvider, setCustomProvider] =
    useState<ModelOption["provider"]>("openai");
  const [rightPadding, setRightPadding] = useState(32);
  const commandRef = useRef<HTMLDivElement>(null);
  const inputSuffixRef = useRef<HTMLDivElement>(null);

  const closeDropdown = useCallback(() => {
    setOpen(false);
    setSearchValue(null);
  }, []);

  useEffect(() => {
    if (inputSuffixRef.current) {
      const width = inputSuffixRef.current.offsetWidth;
      setRightPadding(width + INPUT_RIGHT_CONTENT_PADDING_PX);
    }
  }, [searchValue]);

  const filteredModels = useMemo(
    () => filterModelsByName(predefinedModels, searchValue ?? ""),
    [searchValue, predefinedModels],
  );

  const hasExactMatch = useMemo(
    () =>
      searchValue !== null &&
      predefinedModels.some(
        (m) => m.displayName.toLowerCase() === searchValue.toLowerCase(),
      ),
    [predefinedModels, searchValue],
  );
  const showCreateOption =
    searchValue !== null && searchValue.length > 0 && !hasExactMatch;

  const handleOpenChange = useCallback(
    (newOpen: boolean) => {
      // Don't close popover when provider Select is open
      if (!newOpen && providerSelectOpen) return;
      setOpen(newOpen);
    },
    [providerSelectOpen],
  );

  const handleKeyDown = useCallback(
    (
      e: React.KeyboardEvent<HTMLInputElement>,
      onChange: (value: ModelOption) => void,
    ) => {
      if (e.key === "Escape") {
        closeDropdown();
        return;
      }

      if (
        e.key === "Enter" &&
        searchValue &&
        showCreateOption &&
        filteredModels.length === 0
      ) {
        e.preventDefault();
        onChange(createCustomModel(searchValue, customProvider));
        closeDropdown();
        return;
      }

      if (COMMAND_MENU_KEYS.includes(e.key)) {
        e.preventDefault();
        if (!open) setOpen(true);
        commandRef.current?.dispatchEvent(
          new KeyboardEvent("keydown", { key: e.key, bubbles: true }),
        );
      }
    },
    [
      showCreateOption,
      filteredModels.length,
      searchValue,
      customProvider,
      closeDropdown,
      open,
    ],
  );

  return (
    <FormField
      control={control}
      name="model"
      render={({ field }) => {
        const selectedIsCustom = isCustomModel(field.value, predefinedModels);
        const inputValue = searchValue ?? field.value?.displayName ?? "";

        const handleSelect = (model: ModelOption) => {
          field.onChange(model);
          closeDropdown();
        };

        const handleSelectCustom = () => {
          if (!searchValue) return;
          field.onChange(createCustomModel(searchValue, customProvider));
          closeDropdown();
        };

        const handleBlur = (e: React.FocusEvent) => {
          if (isFocusWithinPopover(e.relatedTarget as Element | null)) {
            return;
          }

          if (searchValue === "") {
            field.onChange(undefined);
            closeDropdown();
            return;
          }

          if (!showCreateOption || !searchValue) return;

          field.onChange(createCustomModel(searchValue, customProvider));
          closeDropdown();
        };

        return (
          <FormItem>
            <FormLabel>Model</FormLabel>
            <div className="grid gap-x-8 md:grid-cols-2">
              <div className="w-full space-y-2">
                <Popover open={open} onOpenChange={handleOpenChange}>
                  <PopoverAnchor asChild>
                    <div className="group relative">
                      <Input
                        placeholder="Select model..."
                        value={inputValue}
                        onChange={(e) => {
                          setSearchValue(e.target.value);
                          if (!open) setOpen(true);
                        }}
                        onFocus={() => setOpen(true)}
                        onClick={() => setOpen(true)}
                        onBlur={handleBlur}
                        onKeyDown={(e) => handleKeyDown(e, field.onChange)}
                        style={{ paddingRight: rightPadding }}
                        className={clsx(
                          "border-border placeholder:text-fg-secondary hover:border-border-accent hover:bg-bg-primary focus-visible:border-border-accent focus-visible:ring-0",
                          open && "border-border-accent",
                        )}
                      />
                      <div
                        ref={inputSuffixRef}
                        className="pointer-events-none absolute top-1/2 right-3 flex -translate-y-1/2 items-center gap-2"
                      >
                        {field.value && searchValue === null && (
                          <ModelBadge provider={field.value.provider} />
                        )}
                        <ChevronDown
                          className={clsx(
                            "text-fg-muted group-hover:text-fg-tertiary h-4 w-4 shrink-0",
                            open && "-rotate-180",
                          )}
                        />
                      </div>
                    </div>
                  </PopoverAnchor>

                  <PopoverContent
                    className="flex w-[var(--radix-popover-trigger-width)] flex-col p-0"
                    align="start"
                    onOpenAutoFocus={(e) => e.preventDefault()}
                    onInteractOutside={(e) => {
                      const target = e.target as Element | null;
                      if (target?.closest(RADIX_SELECT_SELECTOR)) {
                        e.preventDefault();
                      }
                    }}
                  >
                    <Command ref={commandRef} shouldFilter={false}>
                      <CommandList>
                        {filteredModels.length === 0 &&
                          !showCreateOption &&
                          !selectedIsCustom && (
                            <CommandEmpty>No models found</CommandEmpty>
                          )}

                        {showCreateOption && (
                          <>
                            <CommandGroup heading="Custom">
                              <CommandItem
                                value={`custom::${searchValue}`}
                                onSelect={handleSelectCustom}
                                className="flex items-center justify-between gap-2"
                              >
                                <span className="min-w-0 truncate font-mono text-sm">
                                  {searchValue}
                                </span>
                                <div className="flex shrink-0 items-center gap-2">
                                  <span className="text-muted-foreground text-xs">
                                    Select your provider:
                                  </span>
                                  <ModelProviderSelect
                                    value={customProvider}
                                    onChange={setCustomProvider}
                                    onOpenChange={setProviderSelectOpen}
                                  />
                                </div>
                              </CommandItem>
                            </CommandGroup>
                            {filteredModels.length > 0 && <CommandSeparator />}
                          </>
                        )}

                        {selectedIsCustom && !searchValue && (
                          <>
                            <CommandGroup heading="Custom">
                              <CommandItem
                                value={`custom::${field.value.name}`}
                                onSelect={() => setOpen(false)}
                                className="flex items-center justify-between gap-2"
                              >
                                <span className="min-w-0 truncate font-mono text-sm">
                                  {field.value.displayName}
                                </span>
                                <div className="flex shrink-0 items-center gap-2">
                                  <ModelProviderSelect
                                    value={field.value.provider}
                                    onChange={(p) => {
                                      if (!field.value) return;
                                      field.onChange({
                                        ...field.value,
                                        provider: p,
                                      });
                                    }}
                                    onOpenChange={setProviderSelectOpen}
                                  />
                                  <Check className="h-4 w-4" />
                                </div>
                              </CommandItem>
                            </CommandGroup>
                            {filteredModels.length > 0 && <CommandSeparator />}
                          </>
                        )}

                        {providers.map((provider) => {
                          const providerModels = filteredModels.filter(
                            (m) => m.provider === provider,
                          );
                          if (providerModels.length === 0) return null;
                          return (
                            <CommandGroup
                              key={provider}
                              heading={formatProvider(provider).name}
                            >
                              {providerModels.map((model) => (
                                <CommandItem
                                  key={`${model.provider}::${model.name}`}
                                  value={`${model.provider}::${model.name}`}
                                  onSelect={() => handleSelect(model)}
                                  className="flex items-center justify-between"
                                >
                                  <span className="font-mono text-sm">
                                    {model.displayName}
                                  </span>
                                  <Check
                                    className={clsx(
                                      "h-4 w-4",
                                      isModelSelected(field.value, model)
                                        ? "opacity-100"
                                        : "opacity-0",
                                    )}
                                  />
                                </CommandItem>
                              ))}
                            </CommandGroup>
                          );
                        })}
                      </CommandList>
                    </Command>

                    {!showCreateOption && !selectedIsCustom && (
                      <div className="shrink-0 border-t px-3 py-2">
                        <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
                          <Info className="h-3 w-3 shrink-0" />
                          Type to enter a custom model ID
                        </div>
                      </div>
                    )}
                  </PopoverContent>
                </Popover>
              </div>
            </div>
          </FormItem>
        );
      }}
    />
  );
}
