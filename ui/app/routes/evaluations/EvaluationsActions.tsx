import { ActionBar } from "~/components/layout/ActionBar";
import { NewRunButton } from "./NewRunButton";

interface EvaluationsActionsProps {
  onNewRun: () => void;
}

export function EvaluationsActions({ onNewRun }: EvaluationsActionsProps) {
  return (
    <ActionBar>
      <NewRunButton onClick={onNewRun} />
    </ActionBar>
  );
}
