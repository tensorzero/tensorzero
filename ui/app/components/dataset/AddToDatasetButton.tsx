import { useEffect, useState } from "react";
import { useFetcher, Link } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
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

  // Handle success/error states from the fetcher
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.error) {
        toast({
          title: "Failed to add to dataset",
          description: fetcher.data.error,
          variant: "destructive",
        });
      } else if (fetcher.data.redirectTo) {
        toast({
          title: "New Datapoint",
          description: "A datapoint was created successfully.",
          action: (
            <ToastAction altText="View Datapoint" asChild>
              <Link to={fetcher.data.redirectTo}>View</Link>
            </ToastAction>
          ),
        });
      }
    }
  }, [fetcher.state, fetcher.data, toast]);

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
      formData.append("_action", "addToDataset");
      fetcher.submit(formData, { method: "post" });
    }
  };

  // Handle the output selection from the alert dialog
  const handleOutputSelect = (output: "inherit" | "demonstration" | "none") => {
    handleDatasetAction(selectedDataset, output);
    setOutputDialogOpen(false);
  };

  const datasetSelector = (
    <DatasetSelector
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
      buttonProps={{
        size: "sm",
      }}
      disabled={isReadOnly}
    />
  );

  const alertDialog = (
    <AlertDialog open={outputDialogOpen} onOpenChange={setOutputDialogOpen}>
      <AlertDialogContent className="min-w-[600px]">
        <AlertDialogHeader>
          <AlertDialogTitle>
            What should be the datapoint's output?
          </AlertDialogTitle>
          <AlertDialogDescription>
            Each datapoint includes an optional output field. The choice should
            depend on your use case. For example, you might prefer
            demonstrations for fine-tuning.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="flex justify-center gap-4">
          <AlertDialogCancel onClick={() => setOutputDialogOpen(false)}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => handleOutputSelect("inherit")}>
            Inference Output
          </AlertDialogAction>
          {hasDemonstration && (
            <AlertDialogAction
              onClick={() => handleOutputSelect("demonstration")}
            >
              Demonstration
            </AlertDialogAction>
          )}
          <AlertDialogAction onClick={() => handleOutputSelect("none")}>
            None
          </AlertDialogAction>
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
