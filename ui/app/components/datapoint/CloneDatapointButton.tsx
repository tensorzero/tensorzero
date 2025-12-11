import { useEffect, useState } from "react";
import { useFetcher, Link } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { useToast } from "~/hooks/use-toast";
import { ToastAction } from "~/components/ui/toast";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import type { Datapoint } from "~/types/tensorzero";

export interface CloneDatapointButtonProps {
  datapoint: Datapoint;
}

export function CloneDatapointButton({ datapoint }: CloneDatapointButtonProps) {
  const [selectedDataset, setSelectedDataset] = useState("");
  const fetcher = useFetcher();
  const { toast } = useToast();
  const isReadOnly = useReadOnly();

  // Handle success/error states from the fetcher
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.error) {
        const { dismiss } = toast.error({
          title: "Failed to clone datapoint",
          description: fetcher.data.error,
        });
        return () => dismiss({ immediate: true });
      } else if (fetcher.data.redirectTo) {
        const { dismiss } = toast.success({
          title: "Datapoint Cloned",
          description: "The datapoint was cloned successfully.",
          action: (
            <ToastAction altText="View Datapoint" asChild>
              <Link to={fetcher.data.redirectTo}>View</Link>
            </ToastAction>
          ),
        });
        setSelectedDataset("");
        return () => dismiss({ immediate: true });
      }
    }
    return;
  }, [fetcher.state, fetcher.data, toast]);

  const handleDatasetSelect = (dataset: string) => {
    setSelectedDataset(dataset);

    const formData = new FormData();
    formData.append("action", "clone");
    formData.append("target_dataset", dataset);
    formData.append("datapoint", JSON.stringify(datapoint));

    fetcher.submit(formData, { method: "post" });
  };

  const datasetSelector = (
    <DatasetSelector
      selected={selectedDataset}
      onSelect={handleDatasetSelect}
      placeholder="Clone to dataset"
      disabled={isReadOnly}
    />
  );

  return <ReadOnlyGuard asChild>{datasetSelector}</ReadOnlyGuard>;
}
