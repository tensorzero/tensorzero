"use client";

import { Slot } from "radix-ui";
import { useReadOnly } from "~/context/read-only";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export interface ReadOnlyGuardProps
  extends React.ComponentPropsWithRef<"button"> {
  asChild?: boolean;
  /**
   * Custom tooltip message to show when disabled due to read-only mode.
   * Defaults to "This feature is not available in read-only mode."
   */
  readOnlyTooltip?: string;
}

export function ReadOnlyGuard({
  asChild,
  readOnlyTooltip = "This feature is not available in read-only mode.",
  ...props
}: ReadOnlyGuardProps) {
  const Component = asChild ? Slot.Root : "button";
  const isReadOnly = useReadOnly();
  const isDisabled = isReadOnly || props.disabled;

  // If not in read-only mode, just render the component normally
  if (!isReadOnly) {
    return <Component {...props} disabled={props.disabled} />;
  }

  // When disabled due to read-only mode, wrap with tooltip
  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <span className="inline-block">
            <Component {...props} disabled={isDisabled} />
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p>{readOnlyTooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
