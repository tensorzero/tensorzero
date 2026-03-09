import { HoverCard } from "radix-ui";
import { cn } from "~/utils/common";

interface HoverPopoverProps {
  trigger: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  align?: "start" | "center" | "end";
  sideOffset?: number;
  className?: string;
  openDelay?: number;
  closeDelay?: number;
  onOpenChange?: (open: boolean) => void;
}

export function HoverPopover({
  trigger,
  children,
  side = "top",
  align = "center",
  sideOffset = 4,
  className,
  openDelay = 0,
  closeDelay = 150,
  onOpenChange,
}: HoverPopoverProps) {
  return (
    <HoverCard.Root
      openDelay={openDelay}
      closeDelay={closeDelay}
      onOpenChange={onOpenChange}
    >
      <HoverCard.Trigger asChild>{trigger}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side={side}
          align={align}
          sideOffset={sideOffset}
          className={cn(
            "bg-popover text-popover-foreground z-50 rounded-md border p-3 shadow-md",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
            className,
          )}
        >
          {children}
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}
