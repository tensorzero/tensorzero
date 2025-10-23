import { ActionBar } from "~/components/layout/ActionBar";
import { NewRunButton } from "./NewRunButton";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface EvaluationsActionsProps {
  onNewRun: () => void;
}

export function EvaluationsActions({ onNewRun }: EvaluationsActionsProps) {
  return (
    <ActionBar>
      <ReadOnlyGuard asChild>
        <NewRunButton onClick={onNewRun} />
      </ReadOnlyGuard>
    </ActionBar>
  );
}
