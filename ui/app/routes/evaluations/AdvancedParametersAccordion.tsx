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
import { AdaptiveStoppingPrecision } from "./AdaptiveStoppingPrecision";

export interface AdvancedParametersAccordionProps {
  inferenceCache: InferenceCacheSetting;
  setInferenceCache: (inference_cache: InferenceCacheSetting) => void;
  precisionTargets: Record<string, string>;
  setPrecisionTargets: (value: Record<string, string>) => void;
  arePrecisionTargetsValid: boolean;
  evaluatorNames: string[];
  defaultOpen?: boolean;
}

export function AdvancedParametersAccordion({
  inferenceCache,
  setInferenceCache,
  precisionTargets,
  setPrecisionTargets,
  arePrecisionTargetsValid: _arePrecisionTargetsValid,
  evaluatorNames,
  defaultOpen,
}: AdvancedParametersAccordionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen ?? false);

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
            <AdaptiveStoppingPrecision
              precisionTargets={precisionTargets}
              setPrecisionTargets={setPrecisionTargets}
              evaluatorNames={evaluatorNames}
            />
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
