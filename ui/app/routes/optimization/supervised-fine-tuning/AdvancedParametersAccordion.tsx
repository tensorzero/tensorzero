import { useEffect, useState } from "react";
import type { Control } from "react-hook-form";
import { useFormState } from "react-hook-form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { HelpTooltip, docsUrl } from "~/components/ui/HelpTooltip";
import type { SFTFormValues } from "./types";
import { NumberInputWithButtons } from "~/components/utils/NumberInputWithButtons";

type AdvancedParametersAccordionProps = {
  control: Control<SFTFormValues>;
  maxSamplesLimit?: number;
};

export function AdvancedParametersAccordion({
  control,
  maxSamplesLimit,
}: AdvancedParametersAccordionProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { errors } = useFormState({
    control,
  });

  const hasAdvancedErrors = Boolean(
    errors.validationSplitPercent || errors.maxSamples,
  );

  useEffect(() => {
    if (hasAdvancedErrors) {
      setIsOpen(true);
    }
  }, [hasAdvancedErrors]);

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
            <FormField
              control={control}
              name="validationSplitPercent"
              render={({ field }) => (
                <FormItem>
                  <div className="flex items-center gap-1.5">
                    <FormLabel>Validation Split (%)</FormLabel>
                    <HelpTooltip
                      link={{
                        href: docsUrl("optimization/supervised-fine-tuning"),
                      }}
                    >
                      Data reserved for detecting overfitting during
                      fine-tuning.
                    </HelpTooltip>
                  </div>
                  <div className="grid grid-cols-2 gap-x-8">
                    <div className="flex flex-col gap-2">
                      <NumberInputWithButtons
                        value={field.value}
                        onChange={field.onChange}
                        min={0}
                        max={100}
                        aria-label-increase="Increase validation split percentage by 1"
                        aria-label-decrease="Decrease validation split percentage by 1"
                      />
                      {errors.validationSplitPercent && (
                        <p className="text-xs text-red-500">
                          {errors.validationSplitPercent.message}
                        </p>
                      )}
                    </div>
                  </div>
                </FormItem>
              )}
            />

            <FormField
              control={control}
              name="maxSamples"
              render={({ field }) => (
                <FormItem>
                  <div className="flex items-center gap-1.5">
                    <FormLabel>Max. Samples</FormLabel>
                    <HelpTooltip
                      link={{
                        href: docsUrl("optimization/supervised-fine-tuning"),
                      }}
                    >
                      Limited by the number of curated inferences that meet the
                      metric threshold.
                    </HelpTooltip>
                  </div>
                  <div className="grid grid-cols-2 gap-x-8">
                    <div className="flex flex-col gap-1">
                      <NumberInputWithButtons
                        value={field.value || null}
                        onChange={field.onChange}
                        min={10}
                        max={maxSamplesLimit}
                        aria-label-increase="Increase max samples by 1"
                        aria-label-decrease="Decrease max samples by 1"
                      />
                      {errors.maxSamples && (
                        <p className="text-xs text-red-500">
                          {errors.maxSamples.message}
                        </p>
                      )}
                    </div>
                  </div>
                </FormItem>
              )}
            />
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
