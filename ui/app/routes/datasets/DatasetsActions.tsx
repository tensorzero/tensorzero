import { ActionBar } from "~/components/layout/ActionBar";
import { BuildDatasetButton } from "./BuildDatasetButton";

interface DatasetsActionsProps {
  onBuildDataset: () => void;
}

export function DatasetsActions({ onBuildDataset }: DatasetsActionsProps) {
  return (
    <ActionBar>
      <BuildDatasetButton onClick={onBuildDataset} />
    </ActionBar>
  );
}
