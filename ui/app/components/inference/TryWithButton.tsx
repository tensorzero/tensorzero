import { useState } from "react";
import { useHydrated } from "~/hooks/use-hydrated";
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
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

export interface TryWithButtonProps {
  options: string[];
  onSelect: (option: string) => void;
  isLoading: boolean;
  isDefaultFunction?: boolean;
}

export function TryWithButton({
  options,
  onSelect,
  isLoading,
  isDefaultFunction,
}: TryWithButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isReadOnly = useReadOnly();
  const isHydrated = useHydrated();
  const isDisabled = isLoading || isReadOnly || !isHydrated;

  return (
    <ReadOnlyGuard asChild>
      <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" disabled={isDisabled}>
            <ButtonIcon as={Compare} variant="tertiary" />
            Try with {isDefaultFunction ? "model" : "variant"}
            <ButtonIcon as={ChevronDown} variant="tertiary" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          {options.map((option) => (
            <DropdownMenuItem
              key={option}
              onSelect={() => {
                onSelect(option);
                setIsOpen(false);
              }}
              className="font-mono text-sm"
              disabled={isDisabled}
            >
              {option}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </ReadOnlyGuard>
  );
}
