import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { useFunctionConfig } from "~/context/config";

interface VariantSelectorProps {
  functionName: string | null;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function VariantSelector({
  functionName,
  value,
  onChange,
  disabled,
}: VariantSelectorProps) {
  const functionConfig = useFunctionConfig(functionName ?? "");

  const variants = functionConfig?.variants
    ? Object.keys(functionConfig.variants)
    : [];

  const isPlaceholder = !value || value === "__all__";

  return (
    <Select
      onValueChange={onChange}
      value={value}
      disabled={disabled || !functionName}
    >
      <SelectTrigger
        aria-label="Variant"
        className={isPlaceholder ? "text-muted-foreground" : undefined}
      >
        <SelectValue placeholder="Select a variant" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="__all__" className="text-muted-foreground">
          Select a variant
        </SelectItem>
        {variants.map((name) => (
          <SelectItem key={name} value={name}>
            {name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
