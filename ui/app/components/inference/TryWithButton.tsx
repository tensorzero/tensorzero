import { useState } from "react";
import { Button, ButtonIcon } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { ChevronDown } from "lucide-react";
import { Compare } from "~/components/icons/Icons";
import { useReadOnly } from "~/context/read-only";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export interface TryWithButtonProps {
  options: string[];
  onOptionSelect: (variant: string) => void;
  isLoading: boolean;
  isDefaultFunction?: boolean;
}

export function TryWithButton({
  options: variants,
  onOptionSelect: onVariantSelect,
  isLoading,
  isDefaultFunction,
}: TryWithButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isReadOnly = useReadOnly();
  const isDisabled = isLoading || isReadOnly;

  const button = (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={isDisabled}>
          <ButtonIcon as={Compare} variant="tertiary" />
          Try with {isDefaultFunction ? "model" : "variant"}
          <ButtonIcon as={ChevronDown} variant="tertiary" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        {variants.map((variant) => (
          <DropdownMenuItem
            key={variant}
            onSelect={() => {
              onVariantSelect(variant);
              setIsOpen(false);
            }}
            className="font-mono text-sm"
          >
            {variant}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );

  // Wrap with tooltip when in read-only mode
  if (isReadOnly) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <span className="inline-block">{button}</span>
          </TooltipTrigger>
          <TooltipContent>
            <p>This feature is not available in read-only mode</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return button;
}
