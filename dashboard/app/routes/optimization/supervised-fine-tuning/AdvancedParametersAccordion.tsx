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
import { Input } from "~/components/ui/input";
import type { SFTFormValues } from "./types";

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
    errors.validationSplitPercent || errors.maxSamples || errors.threshold,
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
          <div className="flex items-center gap-2">
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
                  <FormLabel>Validation Split (%)</FormLabel>
                  <div className="grid grid-cols-2 gap-x-8">
                    <div className="flex flex-col gap-2">
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        {...field}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                      />
                      {errors.validationSplitPercent && (
                        <p className="text-sm text-red-500">
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
                  <FormLabel>Max. Samples</FormLabel>
                  <div className="grid grid-cols-2 gap-x-8">
                    <div className="flex flex-col gap-2">
                      <Input
                        type="number"
                        min={10}
                        max={maxSamplesLimit}
                        step={1}
                        {...field}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                      />
                      {errors.maxSamples && (
                        <p className="text-sm text-red-500">
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
