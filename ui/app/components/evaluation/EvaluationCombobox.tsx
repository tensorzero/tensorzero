import { useCallback, useMemo } from "react";
import { Combobox } from "~/components/ui/combobox";
import type { EvaluationConfig } from "~/types/tensorzero";
import { Evaluation } from "~/components/icons/Icons";

type EvaluationComboboxProps = {
  selected: string | null;
  onSelect: (evaluationName: string | null) => void;
  evaluations: { [key: string]: EvaluationConfig | undefined };
  functionName: string | null;
  disabled?: boolean;
};

export function EvaluationCombobox({
  selected,
  onSelect,
  evaluations,
  functionName,
  disabled = false,
}: EvaluationComboboxProps) {
  const filteredEvaluations = useMemo(() => {
    if (!functionName) return [];
    return Object.entries(evaluations)
      .filter(([, config]) => config?.function_name === functionName)
      .map(([name]) => name);
  }, [evaluations, functionName]);

  const handleSelect = useCallback(
    (value: string) => {
      onSelect(value);
    },
    [onSelect],
  );

  const handleClear = useCallback(() => {
    onSelect(null);
  }, [onSelect]);

  const getItemSuffix = useCallback(
    (item: string) => {
      const config = evaluations[item];
      if (!config) return null;
      const evaluatorNames = Object.keys(config.evaluators);
      if (evaluatorNames.length === 0) return null;
      return (
        <span className="text-muted-foreground truncate text-xs">
          ({evaluatorNames.join(", ")})
        </span>
      );
    },
    [evaluations],
  );

  const isDisabled = disabled || filteredEvaluations.length === 0;
  const placeholder =
    filteredEvaluations.length === 0
      ? "No evaluations for this function"
      : "Select evaluation...";

  return (
    <Combobox
      selected={selected}
      onSelect={handleSelect}
      items={filteredEvaluations}
      icon={Evaluation}
      placeholder={placeholder}
      emptyMessage="No evaluations found."
      disabled={isDisabled}
      monospace
      clearable
      onClear={handleClear}
      getItemSuffix={getItemSuffix}
    />
  );
}
