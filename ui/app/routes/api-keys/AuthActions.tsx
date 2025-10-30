import { ActionBar } from "~/components/layout/ActionBar";
import { GenerateKeyButton } from "./GenerateKeyButton";

interface AuthActionsProps {
  onGenerateKey: () => void;
}

export function AuthActions({ onGenerateKey }: AuthActionsProps) {
  return (
    <ActionBar>
      <GenerateKeyButton onClick={onGenerateKey} />
    </ActionBar>
  );
}
