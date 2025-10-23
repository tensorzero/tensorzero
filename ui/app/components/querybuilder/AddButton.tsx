import { Button } from "~/components/ui/button";
import { Plus } from "lucide-react";

interface AddButtonProps {
  label: string;
  onClick?: () => void;
}

export default function AddButton({ label, onClick }: AddButtonProps) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onClick}>
      <Plus className="text-fg-tertiary h-4 w-4" />
      {label}
    </Button>
  );
}
