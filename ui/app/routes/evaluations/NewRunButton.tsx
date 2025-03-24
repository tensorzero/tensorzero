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
      variant="ghost"
      onClick={onClick}
      className={className}
      disabled={disabled}
    >
      <Plus className="mr-2 h-4 w-4" />
      New Run
    </Button>
  );
}
