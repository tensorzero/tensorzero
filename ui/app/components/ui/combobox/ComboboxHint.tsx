import { Info } from "lucide-react";

type ComboboxHintProps = {
  children: React.ReactNode;
};

export function ComboboxHint({ children }: ComboboxHintProps) {
  return (
    <div className="text-muted-foreground flex items-center gap-1.5 border-t px-3 py-2 text-xs">
      <Info className="h-3 w-3 shrink-0" />
      {children}
    </div>
  );
}
