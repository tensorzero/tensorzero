import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { forwardRef, useImperativeHandle, useState } from "react";
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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import { MetricNameWithTooltip } from "./MetricNameWithTooltip";

// Constants
const MAX_NESTING_DEPTH = 2;

// Field-level validation (single source of truth)
const FieldValidation = {
  floatValue: z
    .string()
    .min(1, "Required")
    .refine((val) => !isNaN(parseFloat(val)), "Must be a number")
    .refine((val) => isFinite(parseFloat(val)), "Must be a number"),
  tagKey: z.string().min(1, "Required"),
  tagValue: z.string().min(1, "Required"),
};

// Validate current filter values (used on blur and submit)
function validate(filter: InferenceFilter | undefined): boolean {
  if (!filter) return true;

  if (filter.type === "tag") {
    const keyValid = FieldValidation.tagKey.safeParse(filter.key).success;
    const valueValid = FieldValidation.tagValue.safeParse(filter.value).success;
    return keyValid && valueValid;
  }

  if (filter.type === "float_metric") {
    return FieldValidation.floatValue.safeParse(filter.value.toString())
      .success;
  }

  if (filter.type === "boolean_metric") {
    return true; // Always valid (dropdown)
  }

  if (filter.type === "and" || filter.type === "or") {
    return filter.children.every((child) => validate(child));
  }

  return false; // Unknown filter type
}

// Filter Factory Functions
function createTagFilter(): InferenceFilter {
  return {
    type: "tag",
    key: "",
    value: "",
    comparison_operator: "=",
  };
}

function createMetricFilter(
  metricName: string,
  metricConfig: MetricConfig,
): InferenceFilter | null {
  if (metricConfig.type === "float") {
    return {
      type: "float_metric",
      metric_name: metricName,
      value: 0,
      comparison_operator: ">=",
    };
  } else if (metricConfig.type === "boolean") {
    return {
      type: "boolean_metric",
      metric_name: metricName,
      value: true,
    };
  }
  return null;
}

function createGroupFilter(type: "and" | "or"): InferenceFilter {
  return {
    type,
    children: [],
  };
}

// Interfaces for recursive components
interface FilterNodeProps {
  filter: InferenceFilter;
  onUpdate: (newFilter: InferenceFilter) => void;
  onRemove: () => void;
  depth: number;
  config: ReturnType<typeof useConfig>;
}

// FilterGroup: Renders AND/OR groups
function FilterGroup({
  filter,
  onUpdate,
  onRemove,
  depth,
  config,
}: FilterNodeProps & { filter: InferenceFilter & { type: "and" | "or" } }) {
  const handleToggleOperator = () => {
    const newOperator = filter.type === "and" ? "or" : "and";
    onUpdate({
      ...filter,
      type: newOperator,
    });
  };

  const handleAddChild = (newChild: InferenceFilter) => {
    onUpdate({
      ...filter,
      children: [...filter.children, newChild],
    });
  };

  const handleUpdateChild = (index: number, newChild: InferenceFilter) => {
    const newChildren = [...filter.children];
    newChildren[index] = newChild;
    onUpdate({
      ...filter,
      children: newChildren,
    });
  };

  const handleRemoveChild = (index: number) => {
    const newChildren = filter.children.filter((_, i) => i !== index);

    if (newChildren.length === 0) {
      // Empty group - remove it
      onRemove();
    } else {
      // Keep as group, even with single child
      onUpdate({
        ...filter,
        children: newChildren,
      });
    }
  };

  const handleAddTag = () => {
    handleAddChild(createTagFilter());
  };

  const handleAddMetric = (metricName: string, metricConfig: MetricConfig) => {
    const newFilter = createMetricFilter(metricName, metricConfig);
    if (newFilter) {
      handleAddChild(newFilter);
    }
  };

  const handleAddAndGroup = () => {
    handleAddChild(createGroupFilter("and"));
  };

  const handleAddOrGroup = () => {
    handleAddChild(createGroupFilter("or"));
  };

  const canAddGroup = depth < MAX_NESTING_DEPTH;

  // Background shading based on depth
  const backgroundClass =
    depth === 0 ? "" : depth === 1 ? "bg-muted/40" : "bg-muted/80";

  return (
    <div className="relative">
      {/* AND/OR toggle with X button */}
      <div className="absolute top-1/2 left-0 flex -translate-x-1/2 -translate-y-1/2 -rotate-90 items-center gap-1">
        <TooltipProvider delayDuration={300}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="outline"
                onClick={handleToggleOperator}
                className="hover:text-fg-secondary h-5 cursor-pointer px-1 text-sm font-semibold"
              >
                {filter.type.toUpperCase()}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <span className="text-xs">
                Toggle to {filter.type === "and" ? "OR" : "AND"}
              </span>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <DeleteRowButton onRemove={onRemove} />
      </div>
      <div
        className={`border-border border-l-2 py-5 pr-4 pl-6 ${backgroundClass}`}
      >
        {filter.children.length > 0 && (
          <div className="mb-4 space-y-3">
            {filter.children.map((child, index) => (
              <FilterNodeRenderer
                key={index}
                filter={child}
                onUpdate={(newChild) => handleUpdateChild(index, newChild)}
                onRemove={() => handleRemoveChild(index)}
                depth={depth + 1}
                config={config}
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <MetricSelectorPopover onSelect={handleAddMetric} config={config} />
          <AddButton label="Tag" onClick={handleAddTag} />
          {canAddGroup ? (
            <>
              <AddButton label="And" onClick={handleAddAndGroup} />
              <AddButton label="Or" onClick={handleAddOrGroup} />
            </>
          ) : (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="flex gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled
                      className="cursor-not-allowed"
                    >
                      <Plus className="text-fg-tertiary h-4 w-4" />
                      And
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled
                      className="cursor-not-allowed"
                    >
                      <Plus className="text-fg-tertiary h-4 w-4" />
                      Or
                    </Button>
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <span className="text-xs">
                    Maximum nesting depth reached ({MAX_NESTING_DEPTH} levels)
                  </span>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>
    </div>
  );
}

// FilterNodeRenderer: Recursive renderer for any InferenceFilter
function FilterNodeRenderer({
  filter,
  onUpdate,
  onRemove,
  depth,
  config,
}: FilterNodeProps) {
  // Handle AND/OR groups
  if (filter.type === "and" || filter.type === "or") {
    return (
      <FilterGroup
        filter={filter}
        onUpdate={onUpdate}
        onRemove={onRemove}
        depth={depth}
        config={config}
      />
    );
  }

  // Handle leaf filters
  const handleUpdateFilter = (updates: Record<string, unknown>) => {
    onUpdate({ ...filter, ...updates } as InferenceFilter);
  };

  if (filter.type === "tag") {
    return (
      <TagFilterRow
        filter={filter}
        onUpdate={handleUpdateFilter}
        onRemove={onRemove}
      />
    );
  }

  if (filter.type === "float_metric") {
    const metricConfig = config.metrics[filter.metric_name];
    if (!metricConfig) return null;
    return (
      <FloatMetricFilterRow
        filter={filter}
        metricConfig={metricConfig}
        onUpdate={handleUpdateFilter}
        onRemove={onRemove}
      />
    );
  }

  if (filter.type === "boolean_metric") {
    const metricConfig = config.metrics[filter.metric_name];
    if (!metricConfig) return null;
    return (
      <BooleanMetricFilterRow
        filter={filter}
        metricConfig={metricConfig}
        onUpdate={handleUpdateFilter}
        onRemove={onRemove}
      />
    );
  }

  // Unsupported filter type
  return null;
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

  // Expose validation trigger method to parent via ref
  useImperativeHandle(
    ref,
    () => ({
      triggerValidation: async () => {
        const formValid = await form.trigger();
        const filterValid = validate(inferenceFilter);
        return formValid && filterValid;
      },
    }),
    [inferenceFilter, form],
  );

  const handleAddTag = () => {
    // Wrap in AND group
    setInferenceFilter({
      type: "and",
      children: [createTagFilter()],
    });
  };

  const handleAddMetric = (metricName: string, metricConfig: MetricConfig) => {
    const newFilter = createMetricFilter(metricName, metricConfig);
    if (!newFilter) return; // Unsupported metric type

    // Wrap in AND group
    setInferenceFilter({
      type: "and",
      children: [newFilter],
    });
  };

  const handleAddAnd = () => {
    setInferenceFilter(createGroupFilter("and"));
  };

  const handleAddOr = () => {
    setInferenceFilter(createGroupFilter("or"));
  };

  return (
    <Form {...form}>
      <form className="space-y-6">
        <FunctionFormField control={form.control} />
        <FormLabel>Filter</FormLabel>

        {inferenceFilter ? (
          <div className="py-1">
            <FilterNodeRenderer
              filter={inferenceFilter}
              onUpdate={setInferenceFilter}
              onRemove={() => setInferenceFilter(undefined)}
              depth={0}
              config={config}
            />
          </div>
        ) : (
          <div className="flex gap-2">
            <MetricSelectorPopover onSelect={handleAddMetric} config={config} />
            <AddButton label="Tag" onClick={handleAddTag} />
            <AddButton label="And" onClick={handleAddAnd} />
            <AddButton label="Or" onClick={handleAddOr} />
          </div>
        )}
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
  config: ReturnType<typeof useConfig>;
}

function MetricSelectorPopover({
  onSelect,
  config,
}: MetricSelectorPopoverProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
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

interface DeleteRowButtonProps {
  onRemove: () => void;
}

function DeleteRowButton({ onRemove }: DeleteRowButtonProps) {
  return (
    <Button
      type="button"
      variant="destructiveOutline"
      size="icon"
      onClick={onRemove}
      className="text-fg-tertiary h-5 w-5"
    >
      <X />
    </Button>
  );
}

interface TagFilterRowProps {
  filter: InferenceFilter & { type: "tag" };
  onUpdate: (updates: {
    key?: string;
    value?: string;
    comparison_operator?: "=" | "!=";
  }) => void;
  onRemove: () => void;
}

function TagFilterRow({ filter, onUpdate, onRemove }: TagFilterRowProps) {
  const [keyError, setKeyError] = useState<string>();
  const [valueError, setValueError] = useState<string>();

  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Input
            className={`font-mono ${keyError ? "border-red-500" : ""}`}
            placeholder="tag"
            value={filter.key}
            onChange={(e) => {
              onUpdate({ key: e.target.value });
              setKeyError(undefined);
            }}
            onBlur={() => {
              const result = FieldValidation.tagKey.safeParse(filter.key);
              if (!result.success) {
                setKeyError(result.error.errors[0].message);
              }
            }}
          />
          {keyError && (
            <p className="text-destructive absolute top-1 right-2 text-xs">
              {keyError}
            </p>
          )}
        </div>

        <div className="w-14">
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

        <div className="relative w-48">
          <Input
            className={`font-mono ${valueError ? "border-red-500" : ""}`}
            placeholder="value"
            value={filter.value}
            onChange={(e) => {
              onUpdate({ value: e.target.value });
              setValueError(undefined);
            }}
            onBlur={() => {
              const result = FieldValidation.tagValue.safeParse(filter.value);
              if (!result.success) {
                setValueError(result.error.errors[0].message);
              }
            }}
          />
          {valueError && (
            <p className="text-destructive absolute top-1 right-2 text-xs">
              {valueError}
            </p>
          )}
        </div>

        <DeleteRowButton onRemove={onRemove} />
      </div>
    </div>
  );
}

interface FloatMetricFilterRowProps {
  filter: InferenceFilter & { type: "float_metric" };
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
  const [inputValue, setInputValue] = useState(filter.value.toString());
  const [error, setError] = useState<string>();

  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <MetricNameWithTooltip
            metricName={filter.metric_name}
            metricConfig={metricConfig}
          />
        </div>

        <div className="w-14">
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

        <div className="relative w-48">
          <Input
            type="number"
            step="any"
            placeholder="0"
            value={inputValue}
            onChange={(e) => {
              const val = e.target.value;
              setInputValue(val);
              setError(undefined);

              const parsed = parseFloat(val);
              if (!isNaN(parsed)) {
                onUpdate({ value: parsed });
              }
            }}
            onBlur={() => {
              const result = FieldValidation.floatValue.safeParse(inputValue);

              if (!result.success) {
                setError(result.error.errors[0].message);
              }
            }}
            className={error ? "border-red-500" : ""}
          />
          {error && (
            <p className="text-destructive absolute top-1 right-2 text-xs">
              {error}
            </p>
          )}
        </div>

        <DeleteRowButton onRemove={onRemove} />
      </div>
    </div>
  );
}

interface BooleanMetricFilterRowProps {
  filter: InferenceFilter & { type: "boolean_metric" };
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
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <MetricNameWithTooltip
            metricName={filter.metric_name}
            metricConfig={metricConfig}
          />
        </div>

        <div className="text-fg-secondary flex w-14 items-center justify-center text-sm">
          is
        </div>

        <div className="w-48">
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

        <DeleteRowButton onRemove={onRemove} />
      </div>
    </div>
  );
}
