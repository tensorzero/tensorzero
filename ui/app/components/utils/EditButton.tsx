import { Pencil } from "lucide-react";
import { Button } from "~/components/ui/button";

interface EditButtonProps {
  onClick: () => void;
  className?: string;
}

export function EditButton({ onClick, className }: EditButtonProps) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} className={className}>
      <Pencil className="h-4 w-4" />
    </Button>
  );
}
