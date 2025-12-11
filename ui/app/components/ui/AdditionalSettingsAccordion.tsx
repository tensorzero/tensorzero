import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible";

interface AdditionalSettingsAccordionProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  label?: string;
  children: React.ReactNode;
  className?: string;
}

export function AdditionalSettingsAccordion({
  open,
  onOpenChange,
  label = "Advanced",
  children,
  className,
}: AdditionalSettingsAccordionProps) {
  return (
    <Collapsible
      open={open}
      onOpenChange={onOpenChange}
      className={clsx("w-full", className)}
    >
      <CollapsibleTrigger className="flex cursor-pointer items-center gap-1 text-sm font-medium">
        <span>{label}</span>
        <ChevronDown
          className={clsx(
            "text-muted-foreground h-4 w-4 shrink-0",
            open && "rotate-180",
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down overflow-hidden">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}
