import { useState, memo, useEffect, useMemo } from "react";
import { useFetcher } from "react-router";
import { FormLabel } from "~/components/ui/form";
import { useConfig } from "~/context/config";
import type {
  InferenceFilter,
  MetricConfig,
  MetricsWithFeedbackResponse,
} from "~/types/tensorzero";
import { Button } from "~/components/ui/button";
import { Loader2, Plus } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
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
import { cn } from "~/utils/common";
import {
  TagFilterRow,
  FloatMetricFilterRow,
  BooleanMetricFilterRow,
  DemonstrationFilterRow,
} from "./FilterRows";
import AddButton from "./AddButton";
import { DeleteButton } from "../ui/DeleteButton";

// Constants
const MAX_NESTING_DEPTH = 2;

interface InferenceFilterBuilder {
  inferenceFilter?: InferenceFilter;
  setInferenceFilter: (filter?: InferenceFilter) => void;
  functionName?: string | null;
}

export default function InferenceFilterBuilder({
  inferenceFilter,
  setInferenceFilter,
  functionName,
}: InferenceFilterBuilder) {
  const metricsFetcher = useFetcher<MetricsWithFeedbackResponse>();

  useEffect(() => {
    if (functionName) {
      metricsFetcher.load(
        `/api/function/${encodeURIComponent(functionName)}/feedback_counts`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [functionName]);

  const validMetricNames = useMemo((): Set<string> | null => {
    if (!functionName) return null; // null = show all metrics
    if (!metricsFetcher.data) return new Set<string>(); // empty set while loading
    return new Set<string>(
      metricsFetcher.data.metrics.map((m) => m.metric_name),
    );
  }, [functionName, metricsFetcher.data]);

  const isMetricsLoading =
    functionName !== null &&
    functionName !== undefined &&
    metricsFetcher.state === "loading";

  const handleAddTag = () => {
    // Wrap in AND group
    setInferenceFilter({
      type: "and",
      children: [createTagFilter()],
    });
  };

  const handleAddDemonstration = () => {
    setInferenceFilter({
      type: "and",
      children: [createDemonstrationFilter()],
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
    setInferenceFilter({
      type: "and",
      children: [],
    });
  };

  const handleAddOr = () => {
    setInferenceFilter({
      type: "or",
      children: [],
    });
  };

  return (
    <>
      <FormLabel>Advanced</FormLabel>
      {inferenceFilter ? (
        <div className="py-1 pl-4">
          <FilterNodeRenderer
            filter={inferenceFilter}
            onChange={setInferenceFilter}
            depth={0}
            validMetricNames={validMetricNames}
            isMetricsLoading={isMetricsLoading}
          />
        </div>
      ) : (
        <div className="flex gap-2">
          <AddMetricPopover
            onSelect={handleAddMetric}
            validMetricNames={validMetricNames}
            isLoading={isMetricsLoading}
          />
          <AddButton label="Demonstration" onClick={handleAddDemonstration} />
          <AddButton label="Tag" onClick={handleAddTag} />
          <AddButton label="And" onClick={handleAddAnd} />
          <AddButton label="Or" onClick={handleAddOr} />
        </div>
      )}
    </>
  );
}

// Interfaces for recursive components
interface FilterNodeProps {
  filter: InferenceFilter;
  onChange: (newFilter?: InferenceFilter) => void;
  depth: number;
  validMetricNames?: Set<string> | null;
  isMetricsLoading?: boolean;
}

// FilterGroup: Renders AND/OR groups
const FilterGroup = memo(function FilterGroup({
  filter,
  onChange,
  depth,
  validMetricNames,
  isMetricsLoading,
}: FilterNodeProps & { filter: InferenceFilter & { type: "and" | "or" } }) {
  const handleToggleOperator = () => {
    const newOperator = filter.type === "and" ? "or" : "and";
    onChange({
      ...filter,
      type: newOperator,
    });
  };

  const handleAddChild = (newChild: InferenceFilter) => {
    onChange({
      ...filter,
      children: [...filter.children, newChild],
    });
  };

  const handleUpdateChild = (
    index: number,
    newChild: InferenceFilter | undefined,
  ) => {
    if (newChild === undefined) {
      // Remove child
      const newChildren = filter.children.filter((_, i) => i !== index);
      if (newChildren.length === 0) {
        // Empty group - remove it
        onChange(undefined);
      } else {
        // Keep as group
        onChange({
          ...filter,
          children: newChildren,
        });
      }
    } else {
      // Update child
      const newChildren = [...filter.children];
      newChildren[index] = newChild;
      onChange({
        ...filter,
        children: newChildren,
      });
    }
  };

  const handleAddTag = () => {
    handleAddChild(createTagFilter());
  };

  const handleAddDemonstration = () => {
    handleAddChild(createDemonstrationFilter());
  };

  const handleAddMetric = (metricName: string, metricConfig: MetricConfig) => {
    const newFilter = createMetricFilter(metricName, metricConfig);
    if (newFilter) {
      handleAddChild(newFilter);
    }
  };

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
        <DeleteButton
          onDelete={() => onChange(undefined)}
          label="Delete filter group"
          icon="x"
        />
      </div>
      <div
        className={cn(
          "border-border border-l-2 py-5 pr-4 pl-6",
          depth === 1 && "bg-muted/40",
          depth === 2 && "bg-muted/80",
        )}
      >
        {filter.children.length > 0 && (
          <div className="mb-4 space-y-3">
            {filter.children.map((child, index) => (
              <FilterNodeRenderer
                key={index}
                filter={child}
                onChange={(newChild) => handleUpdateChild(index, newChild)}
                depth={depth + 1}
                validMetricNames={validMetricNames}
                isMetricsLoading={isMetricsLoading}
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <AddMetricPopover
            onSelect={handleAddMetric}
            validMetricNames={validMetricNames}
            isLoading={isMetricsLoading}
          />
          <AddButton label="Tag" onClick={handleAddTag} />
          <AddButton label="Demonstration" onClick={handleAddDemonstration} />
          {depth < MAX_NESTING_DEPTH && (
            <>
              <AddButton
                label="And"
                onClick={() =>
                  handleAddChild({
                    type: "and",
                    children: [],
                  })
                }
              />
              <AddButton
                label="Or"
                onClick={() =>
                  handleAddChild({
                    type: "or",
                    children: [],
                  })
                }
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
});

// MissingMetricError: Displays error when metric is not found
function MissingMetricError({
  metricName,
  onDelete,
}: {
  metricName: string;
  onDelete: () => void;
}) {
  return (
    <div className="flex items-center gap-2 rounded border border-red-300 bg-red-50 px-3 py-2">
      <span className="text-destructive text-sm">
        Unknown Metric: <span className="font-mono">{metricName}</span>
      </span>
      <DeleteButton
        onDelete={onDelete}
        label={`Delete missing metric ${metricName}`}
        icon="x"
      />
    </div>
  );
}

// FilterNodeRenderer: Recursive renderer for any InferenceFilter
const FilterNodeRenderer = memo(function FilterNodeRenderer({
  filter,
  onChange,
  depth,
  validMetricNames,
  isMetricsLoading,
}: FilterNodeProps) {
  const config = useConfig();

  // Handle AND/OR groups
  if (filter.type === "and" || filter.type === "or") {
    return (
      <FilterGroup
        filter={filter}
        onChange={onChange}
        depth={depth}
        validMetricNames={validMetricNames}
        isMetricsLoading={isMetricsLoading}
      />
    );
  }

  // Handle leaf filters
  if (filter.type === "tag") {
    return <TagFilterRow filter={filter} onChange={onChange} />;
  }

  if (filter.type === "float_metric") {
    const metricConfig = config.metrics[filter.metric_name];
    if (!metricConfig) {
      return (
        <MissingMetricError
          metricName={filter.metric_name}
          onDelete={() => onChange(undefined)}
        />
      );
    }
    return (
      <FloatMetricFilterRow
        filter={filter}
        metricConfig={metricConfig}
        onChange={onChange}
      />
    );
  }

  if (filter.type === "boolean_metric") {
    const metricConfig = config.metrics[filter.metric_name];
    if (!metricConfig) {
      return (
        <MissingMetricError
          metricName={filter.metric_name}
          onDelete={() => onChange(undefined)}
        />
      );
    }
    return (
      <BooleanMetricFilterRow
        filter={filter}
        metricConfig={metricConfig}
        onChange={onChange}
      />
    );
  }

  if (filter.type === "demonstration_feedback") {
    return <DemonstrationFilterRow filter={filter} onChange={onChange} />;
  }

  // Unsupported filter type
  return null;
});

interface AddMetricPopoverProps {
  onSelect: (metricName: string, metricConfig: MetricConfig) => void;
  validMetricNames?: Set<string> | null;
  isLoading?: boolean;
}

const AddMetricPopover = memo(function AddMetricPopover({
  onSelect,
  validMetricNames,
  isLoading = false,
}: AddMetricPopoverProps) {
  const config = useConfig();
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const metrics = config.metrics;

  // Filter metrics if validMetricNames is provided
  const filteredMetrics = useMemo(() => {
    const entries = Object.entries(metrics).filter(
      (entry): entry is [string, MetricConfig] => entry[1] !== undefined,
    );

    if (validMetricNames === null || validMetricNames === undefined) {
      // No function selected - show all metrics
      return entries;
    }

    // Function selected - filter to only valid metrics
    return entries.filter(([name]) => validMetricNames.has(name));
  }, [metrics, validMetricNames]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="hover:bg-bg-primary border-border hover:border-border-accent"
          disabled={isLoading}
        >
          {isLoading ? (
            <Loader2 className="text-fg-tertiary h-4 w-4 animate-spin" />
          ) : (
            <Plus className="text-fg-tertiary h-4 w-4" />
          )}
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
            <CommandEmpty>
              {validMetricNames !== null && validMetricNames !== undefined
                ? "No metrics with feedback for this function"
                : "No metrics found"}
            </CommandEmpty>
            <CommandGroup>
              {filteredMetrics.map(([metricName, metricConfig]) => (
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
});

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

function createDemonstrationFilter(): InferenceFilter {
  return {
    type: "demonstration_feedback",
    has_demonstration: true,
  };
}
