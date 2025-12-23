import { ActionBar } from "~/components/layout/ActionBar";
import { BuildDatasetButton } from "./BuildDatasetButton";
import { NewDatapointButton } from "./NewDatapointButton";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface DatasetsActionsProps {
  onBuildDataset: () => void;
  onNewDatapoint: () => void;
}

export function DatasetsActions({
  onBuildDataset,
  onNewDatapoint,
}: DatasetsActionsProps) {
  return (
    <ActionBar>
      <ReadOnlyGuard asChild>
        <BuildDatasetButton onClick={onBuildDataset} />
      </ReadOnlyGuard>
      <ReadOnlyGuard asChild>
        <NewDatapointButton onClick={onNewDatapoint} />
      </ReadOnlyGuard>
    </ActionBar>
  );
}
