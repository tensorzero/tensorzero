import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";

interface BuildDatasetButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function BuildDatasetButton({
  onClick,
  className,
  disabled = false,
}: BuildDatasetButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className={className}
      disabled={disabled}
    >
      <Plus className="mr-2 h-4 w-4 text-fg-tertiary" />
      Build Dataset
    </Button>
  );
}
