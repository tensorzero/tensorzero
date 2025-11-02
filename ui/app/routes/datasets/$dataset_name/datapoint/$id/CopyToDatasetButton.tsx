import { useEffect, useState } from "react";
import { Link, useFetcher } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { useReadOnly } from "~/context/read-only";
import { useToast } from "~/hooks/use-toast";
import { ToastAction } from "~/components/ui/toast";

interface CopyToDatasetButtonProps {
  currentDataset: string;
  datapointId: string;
  functionName: string;
}

type CopyDatapointActionData =
  | {
      success: true;
      redirectTo: string;
      dataset: string;
    }
  | {
      success: false;
      error: string;
    };

export function CopyToDatasetButton({
  currentDataset,
  datapointId,
  functionName,
}: CopyToDatasetButtonProps) {
  const [selectedDataset, setSelectedDataset] = useState("");
  const fetcher = useFetcher<CopyDatapointActionData>();
  const { toast } = useToast();
  const isReadOnly = useReadOnly();

  useEffect(() => {
    if (fetcher.state !== "idle" || !fetcher.data) {
      return;
    }

    if (fetcher.data.success) {
      toast({
        title: "Datapoint copied",
        description: `Copied to dataset ${fetcher.data.dataset}.`,
        action: (
          <ToastAction altText="View datapoint" asChild>
            <Link to={fetcher.data.redirectTo}>View</Link>
          </ToastAction>
        ),
      });
      setSelectedDataset("");
    } else {
      toast({
        title: "Failed to copy datapoint",
        description: fetcher.data.error,
        variant: "destructive",
      });
    }
  }, [fetcher.state, fetcher.data, toast]);

  const isSubmitting =
    fetcher.state === "submitting" || fetcher.state === "loading";

  return (
    <ReadOnlyGuard asChild>
      <DatasetSelector
        selected={selectedDataset}
        onSelect={(dataset) => {
          setSelectedDataset(dataset);
          const formData = new FormData();
          formData.append("_action", "copy_to_dataset");
          formData.append("target_dataset", dataset);
          formData.append("datapoint_id", datapointId);
          fetcher.submit(formData, { method: "post", action: "." });
        }}
        placeholder="Copy to dataset"
        buttonProps={{
          size: "sm",
          disabled: isSubmitting || isReadOnly,
        }}
        disabled={isSubmitting || isReadOnly}
        exclude={[currentDataset]}
        functionName={functionName}
      />
    </ReadOnlyGuard>
  );
}
