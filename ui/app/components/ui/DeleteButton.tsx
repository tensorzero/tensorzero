import { Button } from "~/components/ui/button";
import { Trash2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

// TODO (GabrielBianconi): We could consider unifying this component with:
//
// - ui/app/components/utils/DeleteButton.tsx
// - The delete button in the querybuilder PR
//
// They're all somewhat different so let's snooze for now.

interface DeleteButtonProps {
  label?: string;
  onDelete?: () => void;
}

export function DeleteButton({ label, onDelete }: DeleteButtonProps) {
  return (
    <TooltipProvider>
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
    </TooltipProvider>
  );
}
