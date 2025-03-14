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
import { ChevronUp, ChevronDown } from "lucide-react";

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
                  <FormLabel>Validation Split (%)</FormLabel>
                  <div className="grid grid-cols-2 gap-x-8">
                    <div className="flex flex-col gap-2">
                      <div className="group relative">
                        <Input
                          type="number"
                          min={0}
                          max={100}
                          {...field}
                          onChange={(e) =>
                            field.onChange(Number(e.target.value))
                          }
                          className="[appearance:textfield] focus:ring-2 focus:ring-primary/20 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                        />
                        <div className="absolute bottom-2 right-3 top-2 flex flex-col opacity-0 transition-opacity duration-200 group-focus-within:opacity-100 group-hover:opacity-100">
                          <button
                            type="button"
                            className="flex h-1/2 w-4 cursor-pointer items-center justify-center border-none bg-secondary hover:bg-secondary-foreground/15"
                            onClick={() =>
                              field.onChange(
                                Math.min((field.value || 0) + 1, 100),
                              )
                            }
                            aria-label="Increase validation split percentage by 1"
                          >
                            <ChevronUp className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            className="flex h-1/2 w-4 cursor-pointer items-center justify-center border-none bg-secondary hover:bg-secondary-foreground/15"
                            onClick={() =>
                              field.onChange(
                                Math.max((field.value || 0) - 1, 0),
                              )
                            }
                            aria-label="Decrease validation split percentage by 1"
                          >
                            <ChevronDown className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
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
                      <div className="group relative">
                        <Input
                          type="number"
                          min={10}
                          max={maxSamplesLimit}
                          step={1}
                          {...field}
                          onChange={(e) =>
                            field.onChange(Number(e.target.value))
                          }
                          className="[appearance:textfield] focus:ring-2 focus:ring-primary/20 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                        />
                        <div className="absolute bottom-2 right-3 top-2 flex flex-col opacity-0 transition-opacity duration-200 group-focus-within:opacity-100 group-hover:opacity-100">
                          <button
                            type="button"
                            className="flex h-1/2 w-4 cursor-pointer items-center justify-center border-none bg-secondary hover:bg-secondary-foreground/15"
                            onClick={() =>
                              field.onChange(
                                Math.min(
                                  (field.value || 0) + 1,
                                  maxSamplesLimit || Infinity,
                                ),
                              )
                            }
                            aria-label="Increase max samples by 1"
                          >
                            <ChevronUp className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            className="flex h-1/2 w-4 cursor-pointer items-center justify-center border-none bg-secondary hover:bg-secondary-foreground/15"
                            onClick={() =>
                              field.onChange(
                                Math.max((field.value || 0) - 1, 0),
                              )
                            }
                            aria-label="Decrease max samples by 1"
                          >
                            <ChevronDown className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
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
