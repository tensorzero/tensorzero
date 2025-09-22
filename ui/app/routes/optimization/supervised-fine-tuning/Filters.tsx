import { useDraggable, type DragEndEvent } from "@dnd-kit/core";
import { useFieldArray, type UseFormReturn } from "react-hook-form";
import type { Config } from "tensorzero-node";
import { Placeholder } from "~/components/icons/Icons";
import CurationMetricSelector from "~/components/metric/CurationMetricSelector";
import { useFunctionConfigMetrics } from "~/components/metric/useFunctionConfigMetrics";
import { Button } from "~/components/ui/button";
import {
  ListGroup,
  ListHeader,
  ListProvider,
  type ListItemProps,
} from "~/components/ui/List";
import { type useCountFetcher } from "~/routes/api/curated_inferences/count.route";
import { cn } from "~/utils/common";
import { type SFTFormValues } from "./types";

const ListItemWithHandle = ({
  id,
  name,
  index,
  parent,
  children,
  className,
}: ListItemProps) => {
  const { attributes, listeners, setNodeRef, transform, isDragging } =
    useDraggable({
      id,
      data: { index, parent },
    });

  return (
    <div
      className={cn(
        "bg-background flex items-center gap-3 rounded-md border p-2 shadow-sm",
        className,
      )}
      style={{
        transform: transform
          ? `translateX(${transform.x}px) translateY(${transform.y}px)`
          : "none",
      }}
      ref={setNodeRef}
    >
      <div
        className={cn("cursor-grab", isDragging && "cursor-grabbing")}
        {...listeners}
        {...attributes}
      >
        <Placeholder />
      </div>
      {children ?? <p className="m-0 text-sm font-medium">{name}</p>}
    </div>
  );
};

const metricTemplate = { metric: "", threshold: 0.5 };

export function FiltersInput({
  config,
  form,
  counts,
}: {
  config: Config;
  form: UseFormReturn<SFTFormValues>;
  counts: ReturnType<typeof useCountFetcher>;
}) {
  const {
    formState: { errors },
  } = form;

  // { fields, append, prepend, remove, swap, move, insert }
  const filters = useFieldArray({
    control: form.control,
    name: "filters",
  });

  const functionConfigMetrics = useFunctionConfigMetrics({
    control: form.control,
    functionFieldName: "function",
    config,
    addDemonstrations: true,
  });

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    // filters.swap(1, 2);
  };

  return (
    <div className="flex flex-col">
      <ListProvider onDragEnd={handleDragEnd}>
        <ListGroup id="root">
          <ListHeader>
            <span className="text-sm leading-none font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              Filters
            </span>
          </ListHeader>
          {filters.fields.map(({ id }, i) => {
            const name = ["name", id].join("-");
            return (
              <ListItemWithHandle
                key={name}
                name={name}
                id={name}
                index={i}
                parent={"root"}
              >
                <CurationMetricSelector<SFTFormValues>
                  control={form.control}
                  name={`filters.${i}`}
                  functionConfigMetrics={functionConfigMetrics}
                  feedbackCount={counts.feedbackCount}
                  curatedInferenceCount={counts.curatedInferenceCount}
                  isLoading={counts.isLoading}
                />
              </ListItemWithHandle>
            );
          })}
        </ListGroup>
      </ListProvider>

      <Button
        type="button"
        onClick={() => filters.append({ ...metricTemplate })}
      >
        Append
      </Button>

      {/* <CurationMetricSelector<SFTFormValues>
                control={form.control}
                name="metric"
                functionConfigMetrics={functionConfigMetrics}
                feedbackCount={counts.feedbackCount}
                curatedInferenceCount={counts.curatedInferenceCount}
                isLoading={counts.isLoading}
              /> */}

      {errors.metric && (
        <p className="text-xs text-red-500">{errors.metric.message}</p>
      )}
    </div>
  );
}
