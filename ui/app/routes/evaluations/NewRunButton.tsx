import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";

interface NewRunButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function NewRunButton({
  className,
  disabled = false,
  ...props
}: NewRunButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      className={className}
      disabled={disabled}
      {...props}
    >
      <Plus className="text-fg-tertiary mr-2 h-4 w-4" aria-hidden />
      New Run
    </Button>
  );
}
