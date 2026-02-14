import { AlertTriangle } from "lucide-react";
import { Switch, SwitchSize } from "~/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";

interface YoloModeToggleProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}

export function YoloModeToggle({
  checked,
  onCheckedChange,
}: YoloModeToggleProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <label className="flex cursor-pointer items-center gap-2 select-none">
          <span
            className={cn(
              "flex items-center gap-1.5 text-xs font-medium",
              checked ? "text-orange-500" : "text-fg-muted",
            )}
          >
            {checked && <AlertTriangle className="h-3.5 w-3.5" />}
            YOLO Mode
          </span>
          <Switch
            checked={checked}
            onCheckedChange={onCheckedChange}
            size={SwitchSize.Small}
          />
        </label>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={8} collisionPadding={16}>
        <p className="max-w-48 text-xs">
          Auto-approve all tool calls without review. Use with caution.
        </p>
      </TooltipContent>
    </Tooltip>
  );
}
