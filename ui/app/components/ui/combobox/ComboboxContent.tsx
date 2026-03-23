import { forwardRef } from "react";
import { Command, CommandList, CommandEmpty } from "~/components/ui/command";

type ComboboxContentProps = {
  children: React.ReactNode;
  emptyMessage?: string;
  showEmpty?: boolean;
};

export const ComboboxContent = forwardRef<HTMLDivElement, ComboboxContentProps>(
  function ComboboxContent(
    { children, emptyMessage = "No results found", showEmpty = true },
    ref,
  ) {
    return (
      <Command ref={ref} shouldFilter={false}>
        {/* Stop wheel events from propagating to parent scroll containers
            (e.g. Radix Dialog's body scroll lock) so the list scrolls correctly. */}
        <CommandList onWheel={(e) => e.stopPropagation()}>
          {showEmpty && <CommandEmpty>{emptyMessage}</CommandEmpty>}
          {children}
        </CommandList>
      </Command>
    );
  },
);
