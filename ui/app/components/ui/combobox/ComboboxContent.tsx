import { forwardRef } from "react";
import { Command, CommandEmpty, CommandList } from "~/components/ui/command";

interface ComboboxContentProps {
  children: React.ReactNode;
  emptyMessage?: string;
  showEmpty?: boolean;
}

export const ComboboxContent = forwardRef<HTMLDivElement, ComboboxContentProps>(
  function ComboboxContent(
    { children, emptyMessage = "No results found.", showEmpty = true },
    ref,
  ) {
    return (
      <Command ref={ref} shouldFilter={false}>
        <CommandList>
          {showEmpty && (
            <CommandEmpty className="flex items-center justify-center p-4 text-sm">
              {emptyMessage}
            </CommandEmpty>
          )}
          {children}
        </CommandList>
      </Command>
    );
  },
);
