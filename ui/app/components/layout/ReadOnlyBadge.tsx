/**
 * Read-Only Mode Banner
 *
 * Displays a prominent banner at the top of the application when in read-only mode.
 * Informs users that all write operations are disabled.
 */

import { AlertCircle } from "lucide-react";
import { useReadOnly } from "~/context/read-only";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export function ReadOnlyBadge() {
  const isReadOnly = useReadOnly();

  if (!isReadOnly) {
    return null;
  }

  return (
    <Tooltip delayDuration={200}>
      <TooltipTrigger asChild>
        <Badge
          variant="warning"
          className="mx-1 flex cursor-help items-center gap-2 py-2"
        >
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>Read-only Mode</span>
        </Badge>
      </TooltipTrigger>
      <TooltipContent side="right">
        Features involving inference or database mutation are disabled.
      </TooltipContent>
    </Tooltip>
  );
}
