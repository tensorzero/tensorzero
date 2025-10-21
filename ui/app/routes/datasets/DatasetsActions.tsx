import { ActionBar } from "~/components/layout/ActionBar";
import { BuildDatasetButton } from "./BuildDatasetButton";
import { useReadOnly } from "~/context/read-only";

interface DatasetsActionsProps {
  onBuildDataset: () => void;
}

export function DatasetsActions({ onBuildDataset }: DatasetsActionsProps) {
  const isReadOnly = useReadOnly();
  return (
    <ActionBar>
      <BuildDatasetButton onClick={onBuildDataset} disabled={isReadOnly} />
    </ActionBar>
  );
}
