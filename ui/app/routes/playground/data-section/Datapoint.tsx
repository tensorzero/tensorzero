import { useState } from "react";
import type { VariantConfig } from "tensorzero-node";
import { cn } from "~/utils/common";
import clsx from "clsx";
import Chip from "~/components/ui/Chip";
import AnimatedCollapsible from "../ui/AnimatedCollapsible";
import InputSnippet from "~/components/inference/InputSnippet";
import {
  HEADER_ROW_HEIGHT_CSS_VAR,
  VerticalResizeHandle,
} from "../ui/ResizableQuadrant";
import { RunnableVariant } from "./RunnableVariant";
import { DATAPOINT_GRID_TEMPLATE_ROWS, datapointGrid } from "./layout";
import InferenceOutput from "../ui/InferenceOutput";
import { useDatapoint } from "../queries";
import { Skeleton } from "~/components/ui/skeleton";

function DatapointCell({
  children,
  className,
  ...props
}: React.ComponentProps<"section">) {
  return (
    <section
      {...props}
      className={cn(
        "relative row-span-full grid grid-cols-1 grid-rows-subgrid gap-y-2",
        className,
      )}
    >
      {children}
    </section>
  );
}

export function Datapoint({
  datasetName,
  datapointId,
  variants,
  functionName,
}: {
  datasetName: string;
  datapointId: string;
  variants: readonly (readonly [string, VariantConfig])[];
  functionName: string;
}) {
  const [showInput, setShowInput] = useState(true);
  const [showOutput, setShowOutput] = useState(true);

  const { data: datapoint, isLoading } = useDatapoint(datasetName, datapointId);

  return isLoading || !datapoint ? (
    <Skeleton />
  ) : (
    <section
      className="border-border col-span-full mb-2 grid grid-cols-subgrid gap-y-2 border-b-2 px-3 pt-3 pb-5 last:border-b-0"
      style={{ gridTemplateRows: DATAPOINT_GRID_TEMPLATE_ROWS }}
    >
      <DatapointCell>
        {/* TODO Improve styling for these links */}

        <header
          style={{
            // TODO Fix this
            top: `calc(var(${HEADER_ROW_HEIGHT_CSS_VAR}) + 3rem)`,
          }}
          className={clsx("sticky z-20", datapointGrid({ row: "header" }))}
        >
          <Chip
            label={datapoint.id}
            link={`/datasets/${datasetName}/datapoint/${datapoint.id}`}
            font="mono"
            tooltip="Go to datapoint"
            className="truncate"
          />

          {datapoint.episode_id && (
            <Chip
              label="Episode"
              link={`/observability/episodes/${datapoint.episode_id}`}
              className="truncate"
            />
          )}

          {datapoint.source_inference_id && (
            <Chip
              label="Original inference"
              link={`/observability/inferences/${datapoint.source_inference_id}`}
              className="truncate"
            />
          )}
        </header>

        <AnimatedCollapsible
          isOpen={showInput}
          onOpenChange={setShowInput}
          label="Input"
          className={datapointGrid({ row: "input" })}
        >
          <InputSnippet
            maxHeight="Content"
            system={datapoint.input.system}
            messages={datapoint.input.messages}
          />
        </AnimatedCollapsible>

        <AnimatedCollapsible
          isOpen={showOutput}
          onOpenChange={setShowOutput}
          label="Output"
          className={datapointGrid({ row: "output" })}
        >
          {datapoint.output && <InferenceOutput outputs={datapoint.output} />}
        </AnimatedCollapsible>
      </DatapointCell>

      <VerticalResizeHandle />

      {variants.map(([name, variant]) => (
        <DatapointCell key={name}>
          <RunnableVariant
            functionName={functionName}
            datapoint={datapoint}
            variantName={name}
            variant={variant}
            showInput={showInput}
            showOutput={showOutput}
            setShowInput={setShowInput}
            setShowOutput={setShowOutput}
          />
        </DatapointCell>
      ))}
    </section>
  );
}
