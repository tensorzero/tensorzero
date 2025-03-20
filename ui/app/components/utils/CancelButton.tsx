import { X } from "lucide-react";
import { Button } from "~/components/ui/button";

interface CancelButtonProps {
  onClick: () => void;
  className?: string;
}

export function CancelButton({ onClick, className }: CancelButtonProps) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} className={className}>
      <X className="h-4 w-4" />
    </Button>
  );
}
