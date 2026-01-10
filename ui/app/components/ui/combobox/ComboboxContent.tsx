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
        <CommandList>
          {showEmpty && <CommandEmpty>{emptyMessage}</CommandEmpty>}
          {children}
        </CommandList>
      </Command>
    );
  },
);
