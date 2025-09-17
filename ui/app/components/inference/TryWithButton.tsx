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

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={isLoading}>
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
}
