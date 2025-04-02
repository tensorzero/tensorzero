import { ActionBar } from "~/components/layout/ActionBar";
import { NewRunButton } from "./NewRunButton";

interface EvaluationsActionsProps {
  onNewRun: () => void;
  className?: string;
}

export function EvaluationsActions({
  onNewRun,
  className,
}: EvaluationsActionsProps) {
  return (
    <ActionBar className={className}>
      <NewRunButton onClick={onNewRun} />
    </ActionBar>
  );
}
