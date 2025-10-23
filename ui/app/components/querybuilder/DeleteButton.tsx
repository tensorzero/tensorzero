import { Button } from "~/components/ui/button";
import { X } from "lucide-react";

interface DeleteButtonProps {
  onDelete: () => void;
  ariaLabel?: string;
}

export default function DeleteButton({
  onDelete,
  ariaLabel,
}: DeleteButtonProps) {
  return (
    <Button
      type="button"
      variant="destructiveOutline"
      size="icon"
      onClick={onDelete}
      aria-label={ariaLabel}
      className="text-fg-tertiary h-5 w-5"
    >
      <X />
    </Button>
  );
}
