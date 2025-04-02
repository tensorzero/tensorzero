import { ActionBar } from "~/components/layout/ActionBar";
import { BuildDatasetButton } from "./BuildDatasetButton";

interface DatasetsActionsProps {
  onBuildDataset: () => void;
  className?: string;
}

export function DatasetsActions({
  onBuildDataset,
  className,
}: DatasetsActionsProps) {
  return (
    <ActionBar className={className}>
      <BuildDatasetButton onClick={onBuildDataset} />
    </ActionBar>
  );
}
