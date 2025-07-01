import { useState } from "react";
import type { VariantConfig } from "~/utils/config/variant";
import AnimatedCollapsible from "./ui/AnimatedCollapsible";

/** Each datapoint row has its subgrid: columns inherit from the parent grid, but rows are defined independently. */
export enum DatapointGridRow {
  Input = "input",
  Output = "output",
  Metrics = "metrics",
}

export const DATAPOINT_GRID_TEMPLATE_ROWS = `
[${DatapointGridRow.Input}]   min-content
[${DatapointGridRow.Output}]  min-content
[${DatapointGridRow.Metrics}] 1fr
`;

/**
 * TODO FIX TAILWIND STATIC CLASSES
 * row-[input]
 * row-[output]
 * row-[metrics]
 */

/** Helper to generate the Tailwind class to assign an element to a particular row in the grid layout */
export const setRow = (row: DatapointGridRow) => `row-[${row}]`;

const Datapoint: React.FC<{
  datapointId: string;
  variants: [string, VariantConfig][];
}> = ({ datapointId, variants }) => {
  // TODO Fetch this datapoint (!)

  const [showRenderedInput, setShowRenderedInput] = useState(false);
  const [showGeneratedOutput, setShowGeneratedOutput] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <section
      className="col-span-full grid grid-cols-subgrid"
      style={{ gridTemplateRows: DATAPOINT_GRID_TEMPLATE_ROWS }}
    >
      <div>Datapoint {datapointId}</div>
      {/* TODO link to inference */}
      {/* TODO link to episode */}
      {/* TODO Inputs */}

      {variants.map(([name, variant], i) => (
        <section
          key={name}
          style={{
            gridColumn: i + 2,
          }}
        >
          <section className={setRow(DatapointGridRow.Input)}>
            <AnimatedCollapsible
              isOpen={showRenderedInput}
              onOpenChange={setShowRenderedInput}
              label="Rendered input"
            >
              {/* TODO */}
            </AnimatedCollapsible>
          </section>

          <section className={setRow(DatapointGridRow.Output)}>
            <AnimatedCollapsible
              isOpen={showGeneratedOutput}
              onOpenChange={setShowGeneratedOutput}
              label="Generated output"
            >
              {/* TODO */}
            </AnimatedCollapsible>
          </section>

          <section className={setRow(DatapointGridRow.Metrics)}>
            <AnimatedCollapsible
              isOpen={showMetrics}
              onOpenChange={setShowMetrics}
              label="Metrics"
            >
              {/* TODO */}
            </AnimatedCollapsible>
          </section>
        </section>
      ))}

      {/* TODO Placeholder */}
      {/* <div
          className={clsx(
            "col-span-full bg-orange-200",
            setRow(GridRow.Datapoints),
          )}
        >
          <h1 className="block font-mono text-[10rem] leading-none">
            DATAPOINTS
          </h1>
        </div> */}
    </section>
  );
};

export default Datapoint;
