import { useDatasetCountFetcher } from "~/routes/api/datasets/count_inserts.route";
import type { DatasetBuilderFormValues } from "./types";
import type { Control } from "react-hook-form";
import { useEffect } from "react";
import { useWatch } from "react-hook-form";

export function DatasetCountDisplay({
  control,
  setCountToInsert,
  functionInferenceCount,
  metricFeedbackCount,
  metricCuratedInferenceCount,
}: {
  control: Control<DatasetBuilderFormValues>;
  setCountToInsert: (count: number | null) => void;
  functionInferenceCount: number | null;
  metricFeedbackCount: number | null;
  metricCuratedInferenceCount: number | null;
}) {
  const { count: rowsToInsertCount, isLoading: isLoadingRowsToInsert } =
    useDatasetCountFetcher(control);

  const outputSource = useWatch({
    control,
    name: "output_source",
  });

  useEffect(() => {
    setCountToInsert(rowsToInsertCount);
  }, [rowsToInsertCount, setCountToInsert]);

  if (isLoadingRowsToInsert) {
    return <div>Loading counts...</div>;
  }

  // Determine if any count is available to avoid showing an empty div or only the placeholder
  const hasAnyCountToShow =
    functionInferenceCount !== null ||
    metricFeedbackCount !== null ||
    metricCuratedInferenceCount !== null ||
    rowsToInsertCount !== null;

  return (
    <div className="text-fg-secondary w-full text-sm">
      {functionInferenceCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Function Inferences</span>
          <span className="font-medium">
            {functionInferenceCount.toLocaleString()}
          </span>
        </div>
      )}
      {metricFeedbackCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Metric Feedbacks</span>
          <span className="font-medium">
            {metricFeedbackCount.toLocaleString()}
          </span>
        </div>
      )}
      {metricCuratedInferenceCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Curated Inferences</span>
          <span className="font-medium">
            {metricCuratedInferenceCount.toLocaleString()}
          </span>
        </div>
      )}
      {outputSource === "demonstration" && rowsToInsertCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>
            {metricCuratedInferenceCount !== null
              ? "Curated Inferences with Demonstrations"
              : "Inferences with Demonstrations"}
          </span>
          <span className="font-medium">
            {rowsToInsertCount.toLocaleString()}
          </span>
        </div>
      )}
      {rowsToInsertCount !== null && (
        <div className="text-fg-primary flex flex-row items-center justify-between gap-2 pt-2 font-medium">
          <span>Rows to Insert</span>
          <span className="font-medium">
            {rowsToInsertCount.toLocaleString()}
          </span>
        </div>
      )}
      {!hasAnyCountToShow && !isLoadingRowsToInsert && (
        <div className="flex flex-row items-center justify-between gap-2 pt-2">
          <span>Select criteria to see counts.</span>
        </div>
      )}
    </div>
  );
}
