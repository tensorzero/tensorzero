import { FormField, FormLabel } from "~/components/ui/form";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { HelpTooltip, docsUrl } from "~/components/ui/HelpTooltip";
import type { Control } from "react-hook-form";
import type { DatasetBuilderFormValues } from "./types";

export default function OutputSourceSelector({
  control,
}: {
  control: Control<DatasetBuilderFormValues>;
}) {
  return (
    <FormField
      control={control}
      name="output_source"
      render={({ field }) => (
        <div>
          <div className="flex items-center gap-1.5">
            <FormLabel>Outputs to be used in dataset</FormLabel>
            <HelpTooltip
              link={{
                href: docsUrl("gateway/api-reference/datasets-datapoints"),
              }}
            >
              "None" creates input-only datapoints. "Inference" uses the model
              output. "Demonstration" uses human-provided reference outputs.
            </HelpTooltip>
          </div>
          <div className="mt-2 grid gap-x-8 gap-y-2">
            <RadioGroup onValueChange={field.onChange} value={field.value}>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem value="none" id="none" />
                <FormLabel htmlFor="none">None</FormLabel>
              </div>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem value="inference" id="inference" />
                <FormLabel htmlFor="inference">Inference</FormLabel>
              </div>
              <div className="flex h-5 items-center space-x-2">
                <RadioGroupItem value="demonstration" id="demonstration" />
                <FormLabel htmlFor="demonstration">Demonstration</FormLabel>
              </div>
            </RadioGroup>
          </div>
        </div>
      )}
    />
  );
}
