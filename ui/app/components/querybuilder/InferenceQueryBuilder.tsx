import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { forwardRef, useImperativeHandle, useMemo, useState } from "react";
import { Form } from "~/components/ui/form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs, useConfig } from "~/context/config";
import type { Control } from "react-hook-form";
import type { InferenceFilter, MetricConfig } from "tensorzero-node";
import { Button } from "~/components/ui/button";
import { Plus, X } from "lucide-react";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";

// Helper type for individual filters
type FilterItem =
  | { type: "tag"; key: string; value: string; comparison_operator: "=" | "!=" }
  | {
      type: "float_metric";
      metric_name: string;
      value: number;
      comparison_operator: "<" | "<=" | "=" | ">" | ">=" | "!=";
    }
  | { type: "boolean_metric"; metric_name: string; value: boolean };

// Parse InferenceFilter into an array of individual filters
function parseInferenceFilter(
  filter: InferenceFilter | undefined,
): FilterItem[] {
  if (!filter) return [];

  if (filter.type === "and") {
    // Flatten AND children
    return filter.children.map((child) => {
      if (child.type === "tag") {
        return {
          type: "tag" as const,
          key: child.key,
          value: child.value,
          comparison_operator: child.comparison_operator,
        };
      }
      if (child.type === "float_metric") {
        return {
          type: "float_metric" as const,
          metric_name: child.metric_name,
          value: child.value,
          comparison_operator: child.comparison_operator,
        };
      }
      if (child.type === "boolean_metric") {
        return {
          type: "boolean_metric" as const,
          metric_name: child.metric_name,
          value: child.value,
        };
      }
      throw new Error(`Unsupported filter type in AND: ${child.type}`);
    });
  }

  if (filter.type === "tag") {
    return [
      {
        type: "tag" as const,
        key: filter.key,
        value: filter.value,
        comparison_operator: filter.comparison_operator,
      },
    ];
  }

  if (filter.type === "float_metric") {
    return [
      {
        type: "float_metric" as const,
        metric_name: filter.metric_name,
        value: filter.value,
        comparison_operator: filter.comparison_operator,
      },
    ];
  }

  if (filter.type === "boolean_metric") {
    return [
      {
        type: "boolean_metric" as const,
        metric_name: filter.metric_name,
        value: filter.value,
      },
    ];
  }

  return [];
}

// Build InferenceFilter from array of individual filters
function buildInferenceFilter(
  filters: FilterItem[],
): InferenceFilter | undefined {
  if (filters.length === 0) return undefined;

  if (filters.length === 1) {
    const filter = filters[0];
    if (filter.type === "tag") {
      return {
        type: "tag",
        key: filter.key,
        value: filter.value,
        comparison_operator: filter.comparison_operator,
      };
    }
    if (filter.type === "float_metric") {
      return {
        type: "float_metric",
        metric_name: filter.metric_name,
        value: filter.value,
        comparison_operator: filter.comparison_operator,
      };
    }
    if (filter.type === "boolean_metric") {
      return {
        type: "boolean_metric",
        metric_name: filter.metric_name,
        value: filter.value,
      };
    }
  }

  // Multiple filters - combine with AND
  return {
    type: "and",
    children: filters.map((filter): InferenceFilter => {
      if (filter.type === "tag") {
        return {
          type: "tag" as const,
          key: filter.key,
          value: filter.value,
          comparison_operator: filter.comparison_operator,
        };
      }
      if (filter.type === "float_metric") {
        return {
          type: "float_metric" as const,
          metric_name: filter.metric_name,
          value: filter.value,
          comparison_operator: filter.comparison_operator,
        };
      }
      if (filter.type === "boolean_metric") {
        return {
          type: "boolean_metric" as const,
          metric_name: filter.metric_name,
          value: filter.value,
        };
      }
      // This should never happen due to type narrowing, but TypeScript needs it
      const _exhaustiveCheck: never = filter;
      throw new Error(
        `Unsupported filter type: ${(_exhaustiveCheck as FilterItem).type}`,
      );
    }),
  };
}

const InferenceQueryBuilderSchema = z.object({
  function: z.string(),
});

export type InferenceQueryBuilderFormValues = z.infer<
  typeof InferenceQueryBuilderSchema
>;

export interface InferenceQueryBuilderRef {
  triggerValidation: () => Promise<boolean>;
}

interface InferenceQueryBuilderProps {
  inferenceFilter?: InferenceFilter;
  setInferenceFilter: React.Dispatch<
    React.SetStateAction<InferenceFilter | undefined>
  >;
}

export const InferenceQueryBuilder = forwardRef<
  InferenceQueryBuilderRef,
  InferenceQueryBuilderProps
>(function InferenceQueryBuilder(
  { inferenceFilter, setInferenceFilter }: InferenceQueryBuilderProps,
  ref,
) {
  const config = useConfig();
  const form = useForm<InferenceQueryBuilderFormValues>({
    defaultValues: {
      function: "",
    },
    resolver: zodResolver(InferenceQueryBuilderSchema),
    mode: "onChange",
  });

  // Parse inferenceFilter into individual filters
  const filters = useMemo(
    () => parseInferenceFilter(inferenceFilter),
    [inferenceFilter],
  );

  // Expose validation trigger method to parent via ref
  useImperativeHandle(ref, () => ({
    triggerValidation: async () => {
      return await form.trigger();
    },
  }));

  const handleAddTag = () => {
    const newFilter: FilterItem = {
      type: "tag",
      key: "",
      value: "",
      comparison_operator: "=",
    };
    setInferenceFilter(buildInferenceFilter([...filters, newFilter]));
  };

  const handleAddMetric = (metricName: string, metricConfig: MetricConfig) => {
    if (metricConfig.type === "float") {
      const newFilter: FilterItem = {
        type: "float_metric",
        metric_name: metricName,
        value: 0,
        comparison_operator: ">=",
      };
      setInferenceFilter(buildInferenceFilter([...filters, newFilter]));
    } else if (metricConfig.type === "boolean") {
      const newFilter: FilterItem = {
        type: "boolean_metric",
        metric_name: metricName,
        value: true,
      };
      setInferenceFilter(buildInferenceFilter([...filters, newFilter]));
    }
  };

  const handleUpdateFilter = (
    index: number,
    updates: Record<string, unknown>,
  ) => {
    const newFilters = [...filters];
    const currentFilter = newFilters[index];

    // Type-safe update based on filter type
    newFilters[index] = { ...currentFilter, ...updates } as FilterItem;

    setInferenceFilter(buildInferenceFilter(newFilters));
  };

  const handleRemoveFilter = (index: number) => {
    const newFilters = filters.filter((_, i) => i !== index);
    setInferenceFilter(buildInferenceFilter(newFilters));
  };

  return (
    <Form {...form}>
      <form className="space-y-6">
        <FunctionFormField control={form.control} />

        {filters.length > 0 && (
          <>
            <div className="mb-4">
              <FormLabel>Filters</FormLabel>
            </div>

            <div className="space-y-3">
              {filters.map((filter, index) => {
                if (filter.type === "tag") {
                  return (
                    <TagFilterRow
                      key={index}
                      filter={filter}
                      onUpdate={(updates) => handleUpdateFilter(index, updates)}
                      onRemove={() => handleRemoveFilter(index)}
                    />
                  );
                }
                if (filter.type === "float_metric") {
                  const metricConfig = config.metrics[filter.metric_name];
                  if (!metricConfig) return null;
                  return (
                    <FloatMetricFilterRow
                      key={index}
                      filter={filter}
                      metricConfig={metricConfig}
                      onUpdate={(updates) => handleUpdateFilter(index, updates)}
                      onRemove={() => handleRemoveFilter(index)}
                    />
                  );
                }
                if (filter.type === "boolean_metric") {
                  const metricConfig = config.metrics[filter.metric_name];
                  if (!metricConfig) return null;
                  return (
                    <BooleanMetricFilterRow
                      key={index}
                      filter={filter}
                      metricConfig={metricConfig}
                      onUpdate={(updates) => handleUpdateFilter(index, updates)}
                      onRemove={() => handleRemoveFilter(index)}
                    />
                  );
                }
                return null;
              })}
            </div>
          </>
        )}

        <div className="flex gap-2">
          <MetricSelectorPopover onSelect={handleAddMetric} />
          <AddButton label="Tag" onClick={handleAddTag} />
        </div>
      </form>
    </Form>
  );
});

interface FunctionFormFieldProps {
  control: Control<InferenceQueryBuilderFormValues>;
}

function FunctionFormField({ control }: FunctionFormFieldProps) {
  const functions = useAllFunctionConfigs();

  return (
    <FormField
      control={control}
      name="function"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Function</FormLabel>
          <FunctionSelector
            selected={field.value}
            onSelect={(value) => {
              field.onChange(value);
            }}
            functions={functions}
          />
        </FormItem>
      )}
    />
  );
}

interface MetricSelectorPopoverProps {
  onSelect: (metricName: string, metricConfig: MetricConfig) => void;
}

function MetricSelectorPopover({ onSelect }: MetricSelectorPopoverProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const config = useConfig();
  const metrics = config.metrics;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="hover:bg-bg-primary border-border hover:border-border-accent"
        >
          <Plus className="text-fg-tertiary h-4 w-4" />
          Metric
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[400px] p-0" align="start">
        <Command>
          <CommandInput
            placeholder="Find a metric..."
            value={inputValue}
            onValueChange={setInputValue}
            className="h-9"
          />
          <CommandList>
            <CommandEmpty className="px-4 py-2 text-sm">
              No metrics found.
            </CommandEmpty>
            <CommandGroup>
              {Object.entries(metrics)
                .filter(
                  (entry): entry is [string, MetricConfig] =>
                    entry[1] !== undefined,
                )
                .map(([metricName, metricConfig]) => (
                  <CommandItem
                    key={metricName}
                    value={metricName}
                    onSelect={() => {
                      onSelect(metricName, metricConfig);
                      setInputValue("");
                      setOpen(false);
                    }}
                    className="group flex w-full cursor-pointer items-center justify-between gap-2"
                  >
                    <span className="truncate font-mono">{metricName}</span>
                    <FeedbackBadges metric={metricConfig} showLevel={false} />
                  </CommandItem>
                ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

interface AddButtonProps {
  label: string;
  onClick?: () => void;
}

function AddButton({ label, onClick }: AddButtonProps) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onClick}>
      <Plus className="text-fg-tertiary h-4 w-4" />
      {label}
    </Button>
  );
}

interface TagFilterRowProps {
  filter: FilterItem & { type: "tag" };
  onUpdate: (updates: {
    key?: string;
    value?: string;
    comparison_operator?: "=" | "!=";
  }) => void;
  onRemove: () => void;
}

function TagFilterRow({ filter, onUpdate, onRemove }: TagFilterRowProps) {
  return (
    <div>
      <FormLabel>Tag</FormLabel>
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <Input
            className="font-mono"
            value={filter.key}
            onChange={(e) => onUpdate({ key: e.target.value })}
          />
        </div>

        <div className="w-20">
          <Select
            onValueChange={(value) =>
              onUpdate({
                comparison_operator: value as "=" | "!=",
              })
            }
            value={filter.comparison_operator}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="=">=</SelectItem>
              <SelectItem value="!=">≠</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex-1">
          <Input
            className="font-mono"
            value={filter.value}
            onChange={(e) => onUpdate({ value: e.target.value })}
          />
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRemove}
          className="h-8 w-8"
        >
          <X className="text-fg-tertiary h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

interface FloatMetricFilterRowProps {
  filter: FilterItem & { type: "float_metric" };
  metricConfig: MetricConfig;
  onUpdate: (updates: {
    value?: number;
    comparison_operator?: "<" | "<=" | "=" | ">" | ">=" | "!=";
  }) => void;
  onRemove: () => void;
}

function FloatMetricFilterRow({
  filter,
  metricConfig,
  onUpdate,
  onRemove,
}: FloatMetricFilterRowProps) {
  return (
    <div>
      <FormLabel>Metric</FormLabel>
      <div className="flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2">
          <span className="font-mono">{filter.metric_name}</span>
          <FeedbackBadges metric={metricConfig} showLevel={false} />
        </div>

        <div className="w-24">
          <Select
            onValueChange={(value) =>
              onUpdate({
                comparison_operator: value as
                  | "<"
                  | "<="
                  | "="
                  | ">"
                  | ">="
                  | "!=",
              })
            }
            value={filter.comparison_operator}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="<">&lt;</SelectItem>
              <SelectItem value="<=">&le;</SelectItem>
              <SelectItem value="=">=</SelectItem>
              <SelectItem value=">">&gt;</SelectItem>
              <SelectItem value=">=">&ge;</SelectItem>
              <SelectItem value="!=">≠</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="w-32">
          <Input
            type="number"
            step="any"
            placeholder="0.0"
            value={filter.value}
            onChange={(e) => {
              const val = e.target.value;
              const parsed = parseFloat(val);
              if (!isNaN(parsed)) {
                onUpdate({ value: parsed });
              } else if (val === "" || val === "-") {
                onUpdate({ value: 0 });
              }
            }}
          />
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRemove}
          className="h-8 w-8"
        >
          <X className="text-fg-tertiary h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

interface BooleanMetricFilterRowProps {
  filter: FilterItem & { type: "boolean_metric" };
  metricConfig: MetricConfig;
  onUpdate: (updates: { value?: boolean }) => void;
  onRemove: () => void;
}

function BooleanMetricFilterRow({
  filter,
  metricConfig,
  onUpdate,
  onRemove,
}: BooleanMetricFilterRowProps) {
  return (
    <div>
      <FormLabel>Metric</FormLabel>
      <div className="flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2">
          <span className="font-mono">{filter.metric_name}</span>
          <FeedbackBadges metric={metricConfig} showLevel={false} />
        </div>

        <span className="text-fg-secondary text-sm">is</span>

        <div className="w-32">
          <Select
            onValueChange={(value) => onUpdate({ value: value === "true" })}
            value={filter.value.toString()}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true">true</SelectItem>
              <SelectItem value="false">false</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRemove}
          className="h-8 w-8"
        >
          <X className="text-fg-tertiary h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
