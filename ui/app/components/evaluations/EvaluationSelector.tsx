import { Combobox } from "~/components/ui/combobox";
import { Evaluation } from "~/components/icons/Icons";

interface EvaluationSelectorProps {
  selected: string | null;
  onSelect: (evaluationName: string) => void;
  evaluationNames: string[];
  disabled?: boolean;
}

export function EvaluationSelector({
  selected,
  onSelect,
  evaluationNames,
  disabled = false,
}: EvaluationSelectorProps) {
  return (
    <Combobox
      selected={selected}
      onSelect={onSelect}
      items={evaluationNames}
      icon={Evaluation}
      placeholder="Select evaluation"
      emptyMessage="No evaluations found."
      disabled={disabled}
      monospace
    />
  );
}
