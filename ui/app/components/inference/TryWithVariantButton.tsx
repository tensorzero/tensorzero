import { useState } from "react";
import { Button } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { ChevronDown } from "lucide-react";
import { Compare } from "~/components/icons/Icons";

export interface TryWithVariantButtonProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  isLoading: boolean;
}

export function TryWithVariantButton({
  variants,
  onVariantSelect,
  isLoading,
}: TryWithVariantButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={isLoading}>
          <Compare className="text-fg-tertiary h-4 w-4" />
          Try with variant
          <ChevronDown className="text-fg-tertiary h-4 w-4" />
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
