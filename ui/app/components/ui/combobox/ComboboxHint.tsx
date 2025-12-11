import { Info } from "lucide-react";

interface ComboboxHintProps {
  children: React.ReactNode;
}

export function ComboboxHint({ children }: ComboboxHintProps) {
  return (
    <div className="border-t px-3 py-2">
      <p className="text-muted-foreground flex items-center gap-1.5 text-xs">
        <Info className="h-3 w-3 shrink-0" />
        {children}
      </p>
    </div>
  );
}
