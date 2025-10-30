import { ActionBar } from "~/components/layout/ActionBar";
import { GenerateKeyButton } from "./GenerateKeyButton";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface AuthActionsProps {
  onGenerateKey: () => void;
}

export function AuthActions({ onGenerateKey }: AuthActionsProps) {
  return (
    <ActionBar>
      <ReadOnlyGuard asChild>
        <GenerateKeyButton onClick={onGenerateKey} />
      </ReadOnlyGuard>
    </ActionBar>
  );
}
