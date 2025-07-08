import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";

interface NewRunButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function NewRunButton({
  onClick,
  className,
  disabled = false,
}: NewRunButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className={className}
      disabled={disabled}
    >
      <Plus className="text-fg-tertiary mr-2 h-4 w-4" />
      New Run
    </Button>
  );
}
