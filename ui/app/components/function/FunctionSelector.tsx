import { Functions } from "~/components/icons/Icons";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useCallback, useMemo } from "react";
import type { FunctionConfig } from "~/types/tensorzero";
import { Combobox } from "~/components/ui/combobox";

interface FunctionSelectorProps {
  selected: string | null;
  onSelect?: (functionName: string) => void;
  functions: { [x: string]: FunctionConfig | undefined } | undefined;
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
      functions
        ? Object.keys(functions).filter(
            (name) => !(hideDefaultFunction && name === DEFAULT_FUNCTION),
          )
        : [],
    [functions, hideDefaultFunction],
  );

  const getPrefix = useCallback(
    (name: string | null) => {
      if (!name) return <Functions className="h-4 w-4 shrink-0" />;
      if (!functions) return null;
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
      emptyMessage={
        functions === undefined ? "Config unavailable" : "No functions found"
      }
      ariaLabel={ariaLabel}
      disabled={functions === undefined}
    />
  );
}
