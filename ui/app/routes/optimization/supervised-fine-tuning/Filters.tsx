import { useDraggable, type DragEndEvent } from "@dnd-kit/core";
import {
  useController,
  useWatch,
  type Control,
  type FieldPath,
  type UseFieldArrayReturn,
} from "react-hook-form";
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
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
      {/* <div
        className={cn("cursor-grab", isDragging && "cursor-grabbing")}
        {...listeners}
        {...attributes}
      >
        <Placeholder />
      </div> */}
      {children ?? <p className="m-0 text-sm font-medium">{name}</p>}
    </div>
  );
};

const metricTemplate = { metric: "", threshold: 0.5 };

export function FiltersInput({
  config,
  control,
  filtersArr,
  // @ts-expect-error -- hacking rn sorry
  counts = {},
  names,
}: {
  config: Config;
  control: Control<SFTFormValues>;
  filtersArr: UseFieldArrayReturn<SFTFormValues>;
  counts?: ReturnType<typeof useCountFetcher>;
  names: FieldPath<SFTFormValues>[];
}) {
  // const {
  //   formState: { errors },
  // } = form;

  const { field: logicalOperator } = useController({
    name: "logicalOperator",
    control,
  });

  const filters = useWatch({
    control,
    name: [...names] as const,
  }) as SFTFormValues["filters"][number][];

  const toggleLogicalOperator = () => {
    logicalOperator.onChange(logicalOperator.value === "and" ? "or" : "and");
  };

  const functionConfigMetrics = useFunctionConfigMetrics({
    control,
    functionFieldName: "function",
    config,
    addDemonstrations: true,
  });

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    // filtersArr.swap(1, 2);
  };

  const remove = (index: number) => {
    filtersArr.remove(index);
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
          <div className="group my-1 flex gap-2 border p-2 shadow-sm">
            <div className="flex flex-col items-center">
              {/* <div style={borderStyle} /> */}
              <Button
                type="button"
                className="flex-1 justify-end px-10 [text-orientation:mixed] [writing-mode:sideways-lr]"
                onClick={toggleLogicalOperator}
              >
                <span className="sticky top-2">
                  {logicalOperator.value.toUpperCase()}
                </span>
              </Button>
              {/* <div className="flex-1" style={borderStyle} /> */}
            </div>

            <div className="flex-1">
              {filters.map((_, i) => {
                const name = ["name", i].join("-");
                return (
                  <ListItemWithHandle
                    key={name}
                    name={name}
                    id={name}
                    index={i}
                    parent={"root"}
                  >
                    <CurationMetricSelector<SFTFormValues>
                      control={control}
                      name={`filters.${i}`}
                      functionConfigMetrics={functionConfigMetrics}
                      feedbackCount={counts.feedbackCount}
                      curatedInferenceCount={counts.curatedInferenceCount}
                      isLoading={counts.isLoading}
                    />
                    <Button onClick={() => remove(i)} variant="destructive">
                      <Placeholder />
                    </Button>
                  </ListItemWithHandle>
                );
              })}
            </div>
          </div>
        </ListGroup>
      </ListProvider>

      <Button
        type="button"
        variant={"outline"}
        onClick={() => filtersArr.append({ ...metricTemplate })}
      >
        Append
      </Button>

      {/* <CurationMetricSelector<SFTFormValues>
        control={control}
        name=""
        functionConfigMetrics={functionConfigMetrics}
        feedbackCount={counts.feedbackCount}
        curatedInferenceCount={counts.curatedInferenceCount}
        isLoading={counts.isLoading}
      /> */}

      {/* {errors.metric && (
        <p className="text-xs text-red-500">{errors.metric.message}</p>
      )} */}
    </div>
  );
}
