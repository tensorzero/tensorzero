import { memo } from "react";
import { FormLabel } from "~/components/ui/form";
import type { DatapointFilter } from "~/types/tensorzero";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { TagFilterRow } from "./FilterRows";
import AddButton from "./AddButton";
import { DeleteButton } from "../ui/DeleteButton";

const MAX_NESTING_DEPTH = 2;

interface DatapointFilterBuilderProps {
  datapointFilter?: DatapointFilter;
  setDatapointFilter: (filter: DatapointFilter | undefined) => void;
}

export default function DatapointFilterBuilder({
  datapointFilter,
  setDatapointFilter,
}: DatapointFilterBuilderProps) {
  const handleAddTag = () => {
    setDatapointFilter({
      type: "and",
      children: [createTagFilter()],
    });
  };

  const handleAddAnd = () => {
    setDatapointFilter({
      type: "and",
      children: [],
    });
  };

  const handleAddOr = () => {
    setDatapointFilter({
      type: "or",
      children: [],
    });
  };

  return (
    <>
      <FormLabel>Advanced</FormLabel>
      {datapointFilter ? (
        <div className="py-1 pl-4">
          <FilterNodeRenderer
            filter={datapointFilter}
            onChange={setDatapointFilter}
            depth={0}
          />
        </div>
      ) : (
        <div className="flex gap-2">
          <AddButton label="Tag" onClick={handleAddTag} />
          <AddButton label="And" onClick={handleAddAnd} />
          <AddButton label="Or" onClick={handleAddOr} />
        </div>
      )}
    </>
  );
}

interface FilterNodeProps {
  filter: DatapointFilter;
  onChange: (newFilter: DatapointFilter | undefined) => void;
  depth: number;
}

const FilterGroup = memo(function FilterGroup({
  filter,
  onChange,
  depth,
}: FilterNodeProps & { filter: DatapointFilter & { type: "and" | "or" } }) {
  const handleToggleOperator = () => {
    const newOperator = filter.type === "and" ? "or" : "and";
    onChange({
      ...filter,
      type: newOperator,
    });
  };

  const handleAddChild = (newChild: DatapointFilter) => {
    onChange({
      ...filter,
      children: [...filter.children, newChild],
    });
  };

  const handleUpdateChild = (
    index: number,
    newChild: DatapointFilter | undefined,
  ) => {
    if (newChild === undefined) {
      const newChildren = filter.children.filter((_, i) => i !== index);
      if (newChildren.length === 0) {
        onChange(undefined);
      } else {
        onChange({
          ...filter,
          children: newChildren,
        });
      }
    } else {
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

  return (
    <div className="relative">
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
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <AddButton label="Tag" onClick={handleAddTag} />
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

const FilterNodeRenderer = memo(function FilterNodeRenderer({
  filter,
  onChange,
  depth,
}: FilterNodeProps) {
  if (filter.type === "and" || filter.type === "or") {
    return <FilterGroup filter={filter} onChange={onChange} depth={depth} />;
  }

  if (filter.type === "tag") {
    return <TagFilterRow filter={filter} onChange={onChange} />;
  }

  // Unsupported filter types (time, not) - skip for now
  return null;
});

function createTagFilter(): DatapointFilter {
  return {
    type: "tag",
    key: "",
    value: "",
    comparison_operator: "=",
  };
}
