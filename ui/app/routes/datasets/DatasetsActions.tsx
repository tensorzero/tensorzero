import { ActionBar } from "~/components/layout/ActionBar";
import { BuildDatasetButton } from "./BuildDatasetButton";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface DatasetsActionsProps {
  onBuildDataset: () => void;
}

export function DatasetsActions({ onBuildDataset }: DatasetsActionsProps) {
  return (
    <ActionBar>
      <ReadOnlyGuard asChild>
        <BuildDatasetButton onClick={onBuildDataset} />
      </ReadOnlyGuard>
    </ActionBar>
  );
}
