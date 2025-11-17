import { useState } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Label } from "~/components/ui/label";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import type { InferenceCacheSetting } from "~/utils/evaluations.server";

export interface AdvancedParametersAccordionProps {
  inferenceCache: InferenceCacheSetting;
  setInferenceCache: (inference_cache: InferenceCacheSetting) => void;
  precisionLimits: Record<string, string>;
  setPrecisionLimits: (value: Record<string, string>) => void;
  arePrecisionLimitsValid: boolean;
  evaluatorNames: string[];
  defaultOpen?: boolean;
}

export function AdvancedParametersAccordion({
  inferenceCache,
  setInferenceCache,
  precisionLimits,
  setPrecisionLimits,
  arePrecisionLimitsValid: _arePrecisionLimitsValid,
  evaluatorNames,
  defaultOpen,
}: AdvancedParametersAccordionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen ?? false);

  const handlePrecisionLimitChange = (evaluatorName: string, value: string) => {
    setPrecisionLimits({
      ...precisionLimits,
      [evaluatorName]: value,
    });
  };

  const isPrecisionLimitValid = (value: string): boolean => {
    if (value === "") return true;
    // Check if the entire string is a valid number
    const num = Number(value);
    return !isNaN(num) && num >= 0 && value.trim() !== "";
  };
  return (
    <Accordion
      type="single"
      collapsible
      className="w-full"
      value={isOpen ? "advanced-parameters" : undefined}
      onValueChange={(value) => setIsOpen(value === "advanced-parameters")}
    >
      <AccordionItem value="advanced-parameters">
        <AccordionTrigger className="hover:no-underline">
          <div className="flex items-center gap-1">
            <span>Advanced Parameters</span>
          </div>
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-6 px-3 pt-3">
            <div>
              <Label>Inference Cache</Label>
              <RadioGroup
                value={inferenceCache}
                onValueChange={setInferenceCache}
                className="mt-2 flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="on" id="on" />
                  <Label htmlFor="on">On</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="off" id="off" />
                  <Label htmlFor="off">Off</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="read_only" id="read_only" />
                  <Label htmlFor="read_only">Read Only</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="write_only" id="write_only" />
                  <Label htmlFor="write_only">Write Only</Label>
                </div>
              </RadioGroup>
            </div>
            {evaluatorNames.length > 0 && (
              <div>
                <Label>Precision Limits</Label>
                <p className="text-muted-foreground mb-3 text-xs">
                  Stop running an evaluator when both sides of its 95%
                  confidence interval are within the specified threshold of the
                  mean value. (Set to 0 to disable early stopping)
                </p>
                <div className="space-y-3">
                  {evaluatorNames.map((evaluatorName) => {
                    const value = precisionLimits[evaluatorName] ?? "0.0";
                    const isValid = isPrecisionLimitValid(value);
                    return (
                      <div key={evaluatorName}>
                        <div className="flex items-center gap-3">
                          <Label
                            htmlFor={`precision_limit_${evaluatorName}`}
                            className="text-muted-foreground min-w-[120px] text-xs font-normal"
                          >
                            {evaluatorName}
                          </Label>
                          <input
                            type="text"
                            id={`precision_limit_${evaluatorName}`}
                            name={`precision_limit_${evaluatorName}`}
                            value={value}
                            onChange={(e) =>
                              handlePrecisionLimitChange(
                                evaluatorName,
                                e.target.value,
                              )
                            }
                            placeholder="0.0"
                            className={`border-input bg-background flex-1 rounded-md border px-3 py-2 text-sm ${
                              !isValid
                                ? "border-red-500 focus:ring-red-500"
                                : ""
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
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
