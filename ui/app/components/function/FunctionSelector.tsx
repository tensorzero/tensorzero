import { Functions } from "~/components/icons/Icons";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useCallback, useMemo } from "react";
import type { FunctionConfig } from "~/types/tensorzero";
import { Combobox } from "~/components/ui/combobox";

interface FunctionSelectorProps {
  selected: string | null;
  onSelect?: (functionName: string) => void;
  functions: { [x: string]: FunctionConfig | undefined };
  hideDefaultFunction?: boolean;
  ariaLabel?: string;
}

export function FunctionTypeIcon({ type }: { type: FunctionConfig["type"] }) {
  const iconConfig = getFunctionTypeIcon(type);
  return (
    <div className={`${iconConfig.iconBg} rounded-sm p-0.5`}>
      {iconConfig.icon}
    </div>
  );
}

export function FunctionSelector({
  selected,
  onSelect,
  functions,
  hideDefaultFunction = false,
  ariaLabel,
}: FunctionSelectorProps) {
  const functionNames = useMemo(
    () =>
      Object.keys(functions).filter(
        (name) => !(hideDefaultFunction && name === DEFAULT_FUNCTION),
      ),
    [functions, hideDefaultFunction],
  );

  const getItemIcon = useCallback(
    (name: string | null, _isSelected: boolean) => {
      if (!name) return <Functions className="h-4 w-4 shrink-0" />;
      const fn = functions[name];
      return fn ? <FunctionTypeIcon type={fn.type} /> : null;
    },
    [functions],
  );

  return (
    <Combobox
      selected={selected}
      onSelect={(value) => onSelect?.(value)}
      items={functionNames}
      getItemIcon={getItemIcon}
      placeholder="Select function"
      emptyMessage="No functions found"
      ariaLabel={ariaLabel}
    />
  );
}
