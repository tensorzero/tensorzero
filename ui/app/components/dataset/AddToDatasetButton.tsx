import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useFetcher, Link } from "react-router";
import { DatasetSelect } from "~/components/dataset/DatasetSelect";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from "~/components/ui/alert-dialog";
import { useToast } from "~/hooks/use-toast";
import { ToastAction } from "~/components/ui/toast";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

export interface AddToDatasetButtonProps {
  // Required fields for creating a datapoint
  inferenceId: string;
  functionName: string;
  variantName: string;
  episodeId: string;
  hasDemonstration: boolean;
  // When true, skips the output dialog and always uses "inherit"
  alwaysUseInherit?: boolean;
  // Optional callback for custom behavior
  onDatasetSelect?: (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
  ) => void;
}

export function AddToDatasetButton({
  inferenceId,
  functionName,
  variantName,
  episodeId,
  hasDemonstration,
  alwaysUseInherit = false,
  onDatasetSelect,
}: AddToDatasetButtonProps) {
  const [selectedDataset, setSelectedDataset] = useState("");
  const [outputDialogOpen, setOutputDialogOpen] = useState(false);
  const fetcher = useFetcher();
  const { toast } = useToast();
  const isReadOnly = useReadOnly();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.error) {
        const { dismiss } = toast.error({
          title: "Failed to add to dataset",
          description: fetcher.data.error,
        });
        return () => dismiss({ immediate: true });
      } else if (fetcher.data.redirectTo) {
        queryClient.invalidateQueries({ queryKey: ["DATASETS_COUNT"] });
        const { dismiss } = toast.success({
          title: "New Datapoint",
          description: "A datapoint was created successfully.",
          action: (
            <ToastAction altText="View Datapoint" asChild>
              <Link to={fetcher.data.redirectTo}>View</Link>
            </ToastAction>
          ),
        });
        return () => dismiss({ immediate: true });
      }
    }
    return;
  }, [fetcher.state, fetcher.data, toast, queryClient]);

  // Helper function to handle dataset selection
  const handleDatasetAction = (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
  ) => {
    if (onDatasetSelect) {
      onDatasetSelect(dataset, output);
    } else {
      const formData = new FormData();
      formData.append("dataset", dataset);
      formData.append("output", output);
      formData.append("inference_id", inferenceId);
      formData.append("function_name", functionName);
      formData.append("variant_name", variantName);
      formData.append("episode_id", episodeId);
      fetcher.submit(formData, {
        method: "post",
        action: "/api/datasets/datapoints/from-inference",
      });
    }
  };

  // Handle the output selection from the alert dialog
  const handleOutputSelect = (output: "inherit" | "demonstration" | "none") => {
    handleDatasetAction(selectedDataset, output);
    setOutputDialogOpen(false);
  };

  const datasetSelector = (
    <DatasetSelect
      selected={selectedDataset}
      onSelect={(dataset) => {
        setSelectedDataset(dataset);
        if (alwaysUseInherit) {
          handleDatasetAction(dataset, "inherit");
        } else {
          setOutputDialogOpen(true);
        }
      }}
      placeholder="Add to dataset"
      allowCreation
      disabled={isReadOnly}
    />
  );

  const alertDialog = (
    <AlertDialog open={outputDialogOpen} onOpenChange={setOutputDialogOpen}>
      <AlertDialogContent className="max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle>Datapoint output</AlertDialogTitle>
          <AlertDialogDescription>
            Choose what to use as the output for this datapoint.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="flex flex-col gap-2 py-2">
          <AlertDialogAction
            onClick={() => handleOutputSelect("inherit")}
            className="bg-bg-secondary text-fg-primary hover:bg-bg-tertiary border-border h-auto justify-start rounded-lg border px-4 py-3 text-left font-normal shadow-none"
          >
            <div>
              <div className="text-sm font-medium">Inference Output</div>
              <div className="text-fg-muted mt-0.5 text-xs">
                Use the model&apos;s original response as-is
              </div>
            </div>
          </AlertDialogAction>
          {hasDemonstration && (
            <AlertDialogAction
              onClick={() => handleOutputSelect("demonstration")}
              className="bg-bg-secondary text-fg-primary hover:bg-bg-tertiary border-border h-auto justify-start rounded-lg border px-4 py-3 text-left font-normal shadow-none"
            >
              <div>
                <div className="text-sm font-medium">Demonstration</div>
                <div className="text-fg-muted mt-0.5 text-xs">
                  Use the human-curated demonstration
                </div>
              </div>
            </AlertDialogAction>
          )}
          <AlertDialogAction
            onClick={() => handleOutputSelect("none")}
            className="bg-bg-secondary text-fg-primary hover:bg-bg-tertiary border-border h-auto justify-start rounded-lg border px-4 py-3 text-left font-normal shadow-none"
          >
            <div>
              <div className="text-sm font-medium">None</div>
              <div className="text-fg-muted mt-0.5 text-xs">
                Input only — no output attached
              </div>
            </div>
          </AlertDialogAction>
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setOutputDialogOpen(false)}>
            Cancel
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  return (
    <>
      <ReadOnlyGuard asChild>{datasetSelector}</ReadOnlyGuard>
      {alertDialog}
    </>
  );
}
