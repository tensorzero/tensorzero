import { Combobox } from "~/components/ui/combobox";
import { GitBranch } from "lucide-react";

interface VariantSelectorProps {
  selected: string | null;
  onSelect: (variantName: string) => void;
  variantNames: string[];
  disabled?: boolean;
  placeholder?: string;
}

export function VariantSelector({
  selected,
  onSelect,
  variantNames,
  disabled = false,
  placeholder = "Select variant",
}: VariantSelectorProps) {
  return (
    <Combobox
      selected={selected}
      onSelect={onSelect}
      items={variantNames}
      icon={GitBranch}
      placeholder={placeholder}
      emptyMessage="No variants found."
      disabled={disabled}
      monospace
    />
  );
}
