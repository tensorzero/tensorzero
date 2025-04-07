import { ChevronUp, ChevronDown } from "lucide-react";
import { Input } from "~/components/ui/input";

type NumberInputWithButtonsProps = {
  value: number | null;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  "aria-label-increase"?: string;
  "aria-label-decrease"?: string;
};

export function NumberInputWithButtons({
  value,
  onChange,
  min = 0,
  max = Infinity,
  step = 1,
  "aria-label-increase": ariaLabelIncrease,
  "aria-label-decrease": ariaLabelDecrease,
}: NumberInputWithButtonsProps) {
  return (
    <div className="group relative">
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value === null ? "" : value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="focus:ring-primary/20 [appearance:textfield] focus:ring-2 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
      <div className="absolute top-2 right-3 bottom-2 flex flex-col opacity-0 transition-opacity duration-200 group-focus-within:opacity-100 group-hover:opacity-100">
        <button
          type="button"
          className="bg-secondary hover:bg-secondary-foreground/15 flex h-1/2 w-4 cursor-pointer items-center justify-center border-none"
          onClick={() => onChange(Math.min((value || 0) + step, max))}
          aria-label={ariaLabelIncrease}
        >
          <ChevronUp className="h-3 w-3" />
        </button>
        <button
          type="button"
          className="bg-secondary hover:bg-secondary-foreground/15 flex h-1/2 w-4 cursor-pointer items-center justify-center border-none"
          onClick={() => onChange(Math.max((value || 0) - step, min))}
          aria-label={ariaLabelDecrease}
        >
          <ChevronDown className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
