import { Save } from "lucide-react";
import { Button } from "~/components/ui/button";

interface SaveButtonProps {
  onClick: () => void;
  className?: string;
}

export function SaveButton({ onClick, className }: SaveButtonProps) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} className={className}>
      <Save className="h-4 w-4" />
    </Button>
  );
}
