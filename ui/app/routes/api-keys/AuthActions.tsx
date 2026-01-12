import { ActionBar } from "~/components/layout/ActionBar";
import { GenerateKeyButton } from "./GenerateKeyButton";

interface AuthActionsProps {
  onGenerateKey?: () => void;
  disabled?: boolean;
}

export function AuthActions({ onGenerateKey, disabled }: AuthActionsProps) {
  return (
    <ActionBar>
      <GenerateKeyButton onClick={onGenerateKey} disabled={disabled} />
    </ActionBar>
  );
}
