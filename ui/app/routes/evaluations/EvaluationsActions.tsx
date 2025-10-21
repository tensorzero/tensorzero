import { ActionBar } from "~/components/layout/ActionBar";
import { NewRunButton } from "./NewRunButton";
import { useReadOnly } from "~/context/read-only";

interface EvaluationsActionsProps {
  onNewRun: () => void;
}

export function EvaluationsActions({ onNewRun }: EvaluationsActionsProps) {
  const isReadOnly = useReadOnly();
  return (
    <ActionBar>
      <NewRunButton onClick={onNewRun} disabled={isReadOnly} />
    </ActionBar>
  );
}
