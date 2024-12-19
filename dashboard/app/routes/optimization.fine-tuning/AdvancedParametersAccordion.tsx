import { Control } from "react-hook-form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { Input } from "~/components/ui/input";
import { SFTFormValues } from "./types";

type AdvancedParametersAccordionProps = {
  control: Control<SFTFormValues>;
};

export function AdvancedParametersAccordion({
  control,
}: AdvancedParametersAccordionProps) {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="advanced-parameters">
        <AccordionTrigger className="hover:no-underline">
          <div className="flex items-center gap-2">
            <span>Advanced Parameters</span>
          </div>
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-6 pt-3 px-3">
            <FormField
              control={control}
              name="validationSplitPercent"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Validation Split (%)</FormLabel>
                  <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      {...field}
                      onChange={(e) => field.onChange(Number(e.target.value))}
                    />
                    <div></div>
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
                  <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                    <Input
                      type="number"
                      min={1}
                      step={1}
                      {...field}
                      onChange={(e) => field.onChange(Number(e.target.value))}
                    />
                    <div></div>
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
