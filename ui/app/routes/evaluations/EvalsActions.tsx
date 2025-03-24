import { ActionBar } from "~/components/layout/ActionBar";
import { NewRunButton } from "./NewRunButton";

interface EvalsActionsProps {
  onNewRun: () => void;
  className?: string;
}

export function EvalsActions({ onNewRun, className }: EvalsActionsProps) {
  return (
    <ActionBar className={className}>
      <NewRunButton onClick={onNewRun} />
    </ActionBar>
  );
}
