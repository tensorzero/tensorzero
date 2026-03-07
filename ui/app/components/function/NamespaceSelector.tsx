import { Layers } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { cn } from "~/utils/common";

interface NamespaceSelectorProps {
  namespaces: string[];
  value: string | undefined;
  onChange: (value: string | undefined) => void;
}

export function NamespaceSelector({
  namespaces,
  value,
  onChange,
}: NamespaceSelectorProps) {
  const isFiltered = Boolean(value);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          aria-label="Select Namespace"
          className={cn(
            "focus-visible:ring-0",
            isFiltered &&
              "border-orange-300 bg-orange-50 text-orange-800 hover:bg-orange-100 hover:text-orange-900 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-200 dark:hover:bg-orange-900 dark:hover:text-orange-100",
          )}
        >
          <Layers
            className={cn(
              "h-4 w-4",
              isFiltered && "text-orange-600 dark:text-orange-400",
            )}
          />
          {value ? (
            <span className="font-mono">{value}</span>
          ) : (
            "Select Namespace"
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuCheckboxItem
          checked={!value}
          onCheckedChange={() => onChange(undefined)}
        >
          All
        </DropdownMenuCheckboxItem>
        {namespaces.map((ns) => (
          <DropdownMenuCheckboxItem
            key={ns}
            checked={value === ns}
            onCheckedChange={() => onChange(ns)}
          >
            <span className="font-mono">{ns}</span>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
