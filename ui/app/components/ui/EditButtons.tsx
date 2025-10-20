import { Button } from "~/components/ui/button";
import { Plus, Trash2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface DeleteButtonProps {
  label?: string;
  onDelete?: () => void;
}

export function DeleteButton({ label, onDelete }: DeleteButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="iconSm"
          onClick={onDelete}
          aria-label={label}
          className="text-muted-foreground hover:text-destructive h-6 w-6"
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </TooltipTrigger>
      {label && <TooltipContent>{label}</TooltipContent>}
    </Tooltip>
  );
}

interface AddButtonProps {
  label?: string;
  onAdd?: () => void;
}

export function AddButton({ label, onAdd }: AddButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onAdd}
      className="flex items-center gap-2"
    >
      <Plus className="h-4 w-4" />
      {label}
    </Button>
  );
}
