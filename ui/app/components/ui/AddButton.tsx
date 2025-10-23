import { Button } from "~/components/ui/button";
import { Plus } from "lucide-react";

// TODO (GabrielBianconi): We could consider unifying this component with the add button in the querybuilder PR.

interface AddButtonProps {
  label?: string;
  onAdd?: () => void;
}

export function AddButton({ label, onAdd }: AddButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onAdd}
      className="flex items-center gap-2"
    >
      <Plus className="h-4 w-4" />
      {label}
    </Button>
  );
}
