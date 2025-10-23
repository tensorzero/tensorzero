import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";

interface BuildDatasetButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function BuildDatasetButton({
  disabled = false,
  ...props
}: BuildDatasetButtonProps) {
  return (
    <Button variant="outline" size="sm" disabled={disabled} {...props}>
      <Plus className="text-fg-tertiary mr-2 h-4 w-4" />
      Build Dataset
    </Button>
  );
}
