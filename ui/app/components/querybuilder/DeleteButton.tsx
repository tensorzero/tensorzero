import { Button } from "~/components/ui/button";
import { X } from "lucide-react";

interface DeleteButtonProps {
  onDelete: () => void;
}

export default function DeleteButton({ onDelete }: DeleteButtonProps) {
  return (
    <Button
      type="button"
      variant="destructiveOutline"
      size="icon"
      onClick={onDelete}
      className="text-fg-tertiary h-5 w-5"
    >
      <X />
    </Button>
  );
}
