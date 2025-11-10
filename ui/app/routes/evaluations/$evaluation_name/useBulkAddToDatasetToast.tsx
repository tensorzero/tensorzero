import { useEffect } from "react";
import { Link, useFetcher } from "react-router";
import { ToastAction } from "~/components/ui/toast";
import { useToast } from "~/hooks/use-toast";
import type { SelectedRowData } from "./EvaluationTable";

interface BulkAddToDatasetResponse {
  success: boolean;
  count?: number;
  dataset?: string;
  errors?: string[];
  error?: string;
}

interface UseBulkAddToDatasetToastProps {
  fetcher: ReturnType<typeof useFetcher<BulkAddToDatasetResponse>>;
  toast: ReturnType<typeof useToast>["toast"];
  setSelectedRows: (rows: Map<string, SelectedRowData>) => void;
  setSelectedDataset: (dataset: string) => void;
}

export function useBulkAddToDatasetToast({
  fetcher,
  toast,
  setSelectedRows,
  setSelectedDataset,
}: UseBulkAddToDatasetToastProps) {
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.error) {
        toast.error({
          title: "Failed to add to dataset",
          description: fetcher.data.error,
        });
      } else if (fetcher.data.success) {
        const datasetName = fetcher.data.dataset;
        const hasErrors = fetcher.data.errors && fetcher.data.errors.length > 0;
        const errorCount = hasErrors ? (fetcher.data.errors?.length ?? 0) : 0;

        toast.success({
          title: hasErrors ? "Partially Added to Dataset" : "Added to Dataset",
          description: hasErrors
            ? `${fetcher.data.count} ${fetcher.data.count === 1 ? "inference" : "inferences"} added to: ${datasetName}. ${errorCount} failed to add.`
            : `${fetcher.data.count} ${fetcher.data.count === 1 ? "inference" : "inferences"} added to: ${datasetName}`,
          action: (
            <ToastAction altText="View Dataset" asChild>
              <Link to={`/datasets/${datasetName}`}>View Dataset</Link>
            </ToastAction>
          ),
        });
        setSelectedRows(new Map());
        setSelectedDataset("");
      }
    }
  }, [fetcher.state, fetcher.data, toast, setSelectedRows, setSelectedDataset]);
}
