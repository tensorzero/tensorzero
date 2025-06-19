import { useDatasetCountFetcher } from "~/routes/api/datasets/count_inserts.route";
import type { DatasetBuilderFormValues } from "./types";
import type { Control } from "react-hook-form";

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
  const { count: rowsToInsertCount, isLoading: isLoadingRowsToInsert } = useDatasetCountFetcher(control);

  if (isLoadingRowsToInsert) {
    return <div>Loading counts...</div>;
  }

  setCountToInsert(rowsToInsertCount);

  // Determine if any count is available to avoid showing an empty div or only the placeholder
  const hasAnyCountToShow = 
    functionInferenceCount !== null || 
    metricFeedbackCount !== null || 
    metricCuratedInferenceCount !== null || 
    rowsToInsertCount !== null;

  return (
    <div className="w-full text-sm text-fg-secondary">
      {functionInferenceCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Function Inferences</span>
          <span className="font-medium">{functionInferenceCount.toLocaleString()}</span>
        </div>
      )}
      {metricFeedbackCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Metric Feedbacks</span>
          <span className="font-medium">{metricFeedbackCount.toLocaleString()}</span>
        </div>
      )}
      {metricCuratedInferenceCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Curated Inferences</span>
          <span className="font-medium">{metricCuratedInferenceCount.toLocaleString()}</span>
        </div>
      )}
      {rowsToInsertCount !== null && (
        <div className="flex flex-row items-center pt-2 font-medium text-fg-primary justify-between gap-2">
          <span>Rows to Insert</span>
          <span className="font-medium">{rowsToInsertCount.toLocaleString()}</span>
        </div>
      )}
      {!hasAnyCountToShow && !isLoadingRowsToInsert && (
        <div className="flex flex-row items-center pt-2 justify-between gap-2">
          <span>Select criteria to see counts.</span>
        </div>
      )}
    </div>
  );
}
