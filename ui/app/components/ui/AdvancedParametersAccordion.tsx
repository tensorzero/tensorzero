import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible";

interface AdvancedParametersAccordionProps {
  children: React.ReactNode;
  hasErrors?: boolean;
  label?: string;
  className?: string;
  defaultOpen?: boolean;
}

export function AdvancedParametersAccordion({
  children,
  hasErrors,
  label = "Advanced",
  className,
  defaultOpen = false,
}: AdvancedParametersAccordionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  useEffect(() => {
    if (hasErrors) {
      setIsOpen(true);
    }
  }, [hasErrors]);

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className={clsx("w-full", className)}
    >
      <CollapsibleTrigger className="flex cursor-pointer items-center gap-1 text-sm font-medium">
        <span>{label}</span>
        <ChevronDown
          className={clsx(
            "text-muted-foreground h-4 w-4 shrink-0",
            isOpen && "rotate-180",
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down overflow-hidden">
        <div className="space-y-6 pt-4">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}
