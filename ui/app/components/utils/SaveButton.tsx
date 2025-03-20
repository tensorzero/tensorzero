import { Save } from "lucide-react";
import { Button } from "~/components/ui/button";

interface SaveButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function SaveButton({
  onClick,
  className,
  disabled = false,
}: SaveButtonProps) {
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={onClick}
      className={className}
      disabled={disabled}
    >
      <Save className="h-4 w-4" />
    </Button>
  );
}
