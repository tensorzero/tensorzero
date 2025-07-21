import { useState } from "react";
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

export interface InferenceDatasetButtonProps {
  // Callback receives the chosen dataset name plus a flag indicating if it's new.
  onDatasetSelect: (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
  ) => void;
  hasDemonstration: boolean;
}

export function AddToDatasetButton({
  onDatasetSelect,
  hasDemonstration,
}: InferenceDatasetButtonProps) {
  const [selectedDataset, setSelectedDataset] = useState("");
  const [outputDialogOpen, setOutputDialogOpen] = useState(false);

  // Handle the output selection from the alert dialog
  const handleOutputSelect = (output: "inherit" | "demonstration" | "none") => {
    onDatasetSelect(selectedDataset, output);
    setOutputDialogOpen(false);
  };

  return (
    <>
      <DatasetSelector
        selected={selectedDataset}
        onSelect={(dataset) => {
          setSelectedDataset(dataset);
          setOutputDialogOpen(true);
        }}
        placeholder="Add to dataset"
        buttonProps={{
          size: "sm",
        }}
      />

      <AlertDialog open={outputDialogOpen} onOpenChange={setOutputDialogOpen}>
        <AlertDialogContent className="min-w-[600px]">
          <AlertDialogHeader>
            <AlertDialogTitle>
              What should be the datapoint's output?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Each datapoint includes an optional output field. The choice
              should depend on your use case. For example, you might prefer
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
    </>
  );
}
