import { Layers } from "lucide-react";
import { ButtonSelect } from "~/components/ui/select/ButtonSelect";
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
  const items = ["All", ...namespaces];

  return (
    <ButtonSelect
      items={items}
      selected={value ?? "All"}
      onSelect={(item) => onChange(item === "All" ? undefined : item)}
      searchable={true}
      placeholder="Search namespaces..."
      emptyMessage="No namespaces found"
      triggerClassName={cn(
        "focus-visible:ring-0",
        isFiltered &&
          "border-orange-300 bg-orange-50 text-orange-800 hover:bg-orange-100 hover:text-orange-900 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-200 dark:hover:bg-orange-900 dark:hover:text-orange-100",
      )}
      trigger={
        <>
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
        </>
      }
    />
  );
}
