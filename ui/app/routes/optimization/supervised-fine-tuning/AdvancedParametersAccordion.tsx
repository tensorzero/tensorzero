import { useEffect, useState } from "react";
import type { Control } from "react-hook-form";
import { useFormState } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import type { SFTFormValues } from "./types";
import { NumberInputWithButtons } from "~/components/utils/NumberInputWithButtons";
import { AdditionalSettingsAccordion } from "~/components/ui/AdditionalSettingsAccordion";

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
    <AdditionalSettingsAccordion open={isOpen} onOpenChange={setIsOpen}>
      <div className="space-y-6 pt-4">
        <FormField
          control={control}
          name="validationSplitPercent"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Validation Split (%)</FormLabel>
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
              <FormLabel>Max. Samples</FormLabel>
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
    </AdditionalSettingsAccordion>
  );
}
