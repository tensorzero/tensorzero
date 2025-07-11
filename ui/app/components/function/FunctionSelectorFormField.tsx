import type { Control, Path } from "react-hook-form";
import { type Config } from "tensorzero-node";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { FunctionSelector } from "./FunctionSelector";

type FunctionSelectorFormFieldProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  inferenceCount: number | null;
  config: Config;
  hide_default_function?: boolean;
};

export function FunctionSelectorFormField<T extends Record<string, unknown>>({
  control,
  name,
  config,
  hide_default_function = false,
}: FunctionSelectorFormFieldProps<T>) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel hidden>Function</FormLabel>
          <FunctionSelector
            selected={field.value as string}
            onSelect={field.onChange}
            functions={config.functions}
            hideDefaultFunction={hide_default_function}
          />
        </FormItem>
      )}
    />
  );
}
