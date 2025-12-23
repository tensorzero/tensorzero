import { CircleHelp } from "lucide-react";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export interface AdaptiveStoppingPrecisionProps {
  precisionTargets: Record<string, string>;
  setPrecisionTargets: (value: Record<string, string>) => void;
  evaluatorNames: string[];
}

export function AdaptiveStoppingPrecision({
  precisionTargets,
  setPrecisionTargets,
  evaluatorNames,
}: AdaptiveStoppingPrecisionProps) {
  const handlePrecisionTargetChange = (
    evaluatorName: string,
    value: string,
  ) => {
    setPrecisionTargets({
      ...precisionTargets,
      [evaluatorName]: value,
    });
  };

  const isPrecisionTargetValid = (value: string): boolean => {
    if (value === "") return true;
    // Check if the entire string is a valid number
    const num = Number(value);
    return !isNaN(num) && num >= 0 && value.trim() !== "";
  };

  if (evaluatorNames.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5">
        <Label>Adaptive Stopping Precision</Label>
        <TooltipProvider>
          <Tooltip delayDuration={300}>
            <TooltipTrigger asChild>
              <span className="inline-flex cursor-help">
                <CircleHelp className="text-muted-foreground h-3.5 w-3.5" />
              </span>
            </TooltipTrigger>
            <TooltipContent side="top">
              Stop running an evaluator when both sides of its 95% confidence
              interval are within the specified threshold of the mean value. Set
              to 0 to disable adaptive stopping.
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <div className="space-y-3">
        {evaluatorNames.map((evaluatorName) => {
          const value = precisionTargets[evaluatorName];
          const isValid = isPrecisionTargetValid(value);
          return (
            <div key={evaluatorName}>
              <div className="flex items-center gap-3">
                <Label
                  htmlFor={`precision_target_${evaluatorName}`}
                  className="text-muted-foreground min-w-[120px] text-xs font-normal"
                >
                  {evaluatorName}
                </Label>
                <Input
                  type="text"
                  id={`precision_target_${evaluatorName}`}
                  name={`precision_target_${evaluatorName}`}
                  value={value}
                  onChange={(e) =>
                    handlePrecisionTargetChange(evaluatorName, e.target.value)
                  }
                  placeholder="0.0"
                  className={`flex-1 ${
                    !isValid ? "border-red-500 focus:ring-red-500" : ""
                  }`}
                />
              </div>
              {!isValid && (
                <p className="mt-1 text-xs text-red-500">
                  Must be a non-negative number
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
