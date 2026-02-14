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

export function FunctionTypeBadge({ type }: { type: FunctionConfig["type"] }) {
  const iconConfig = getFunctionTypeIcon(type);
  return (
    <span
      className={`${iconConfig.iconBg} inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 font-mono text-sm`}
    >
      {iconConfig.icon}
      {type}
    </span>
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

  const getPrefix = useCallback(
    (name: string | null) => {
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
      getPrefix={getPrefix}
      placeholder="Select function"
      emptyMessage="No functions found"
      ariaLabel={ariaLabel}
    />
  );
}
