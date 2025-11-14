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
  minInferences: string;
  setMinInferences: (value: string) => void;
  maxInferences: string;
  setMaxInferences: (value: string) => void;
  precisionLimits: string;
  setPrecisionLimits: (value: string) => void;
  defaultOpen?: boolean;
}

export function AdvancedParametersAccordion({
  inferenceCache,
  setInferenceCache,
  minInferences,
  setMinInferences,
  maxInferences,
  setMaxInferences,
  precisionLimits,
  setPrecisionLimits,
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
            <div>
              <Label htmlFor="min_inferences">Min Inferences</Label>
              <p className="text-muted-foreground mb-2 text-xs">
                Minimum samples before checking stopping conditions (default:
                20)
              </p>
              <input
                type="number"
                id="min_inferences"
                name="min_inferences"
                min="1"
                value={minInferences}
                onChange={(e) => setMinInferences(e.target.value)}
                placeholder="20"
                className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
              />
            </div>
            <div>
              <Label htmlFor="max_inferences">Max Inferences</Label>
              <p className="text-muted-foreground mb-2 text-xs">
                Maximum number of datapoints to evaluate (optional)
              </p>
              <input
                type="number"
                id="max_inferences"
                name="max_inferences"
                min="1"
                value={maxInferences}
                onChange={(e) => setMaxInferences(e.target.value)}
                placeholder="No limit"
                className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
              />
            </div>
            <div>
              <Label htmlFor="precision_limits">Precision Limits</Label>
              <p className="text-muted-foreground mb-2 text-xs">
                JSON object mapping evaluator names to CI half-width thresholds
                (e.g., {`{"exact_match": 0.13}`})
              </p>
              <input
                type="text"
                id="precision_limits"
                name="precision_limits"
                value={precisionLimits}
                onChange={(e) => setPrecisionLimits(e.target.value)}
                placeholder="{}"
                className="border-input bg-background w-full rounded-md border px-3 py-2 font-mono text-sm"
              />
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
