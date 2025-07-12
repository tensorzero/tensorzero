import type { VariantConfig } from "tensorzero-node";
import clsx from "clsx";
import { usePlaygroundFunctionAtom } from "../state";
import {
  HEADER_ROW_HEIGHT_CSS_VAR,
  resizableGrid,
} from "../ui/ResizableQuadrant";
import { useCallback } from "react";
import { DatasetSelector } from "../ui/DatasetSelector";
import { useDataset } from "../queries";
import { Skeleton } from "~/components/ui/skeleton";
import { useAtom } from "jotai";
import { Datapoint } from "./Datapoint";

export default function DataSection({
  variants,
  functionName,
}: {
  functionName: string;
  variants: readonly (readonly [string, VariantConfig])[];
}) {
  const functionAtom = usePlaygroundFunctionAtom(functionName);
  const [{ selectedDataset }, setFunctionState] = useAtom(functionAtom);

  const setSelectedDataset = useCallback(
    (datasetName: string | undefined) => {
      setFunctionState((layout) => ({
        ...layout,
        selectedDataset: datasetName,
      }));
    },
    [setFunctionState],
  );

  const datasetQuery = useDataset(selectedDataset, functionName);

  return (
    <section
      className={clsx(
        "relative col-span-full grid grid-cols-subgrid gap-y-3",
        resizableGrid({ row: "content" }),
      )}
    >
      <section
        className="sticky z-10"
        style={{
          top: `calc(var(${HEADER_ROW_HEIGHT_CSS_VAR}) + 1rem)`,
        }}
      >
        {/* TODO Improve dataset selector styling --- appear floating */}
        <DatasetSelector
          className="rounded-2xl border-none shadow-lg"
          allowCreation={false}
          onSelect={setSelectedDataset}
          selected={selectedDataset}
        />
      </section>

      {selectedDataset &&
        (datasetQuery.isFetching || !datasetQuery.data ? (
          <Skeleton />
        ) : datasetQuery.data.length === 0 ? (
          <section className="col-1 row-2 grid py-8">
            <p className="text-muted-foreground text-sm">
              No matching datapoints for selected function
            </p>
          </section>
        ) : (
          datasetQuery.data
            .slice(0, 3) // TODO Remove this constraint. Tons of perf issues with many datapoints - likely due to grids/subgrids or CodeMirror. Consider virtualized list.
            .map((datapoint) => (
              <Datapoint
                datasetName={selectedDataset}
                functionName={functionName}
                key={datapoint.id}
                datapointId={datapoint.id}
                variants={variants}
              />
            ))
        ))}
    </section>
  );
}
