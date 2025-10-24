import type { Control, FieldPath, FieldValues } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { FunctionSelector } from "./FunctionSelector";
import { useAllFunctionConfigs } from "~/context/config";

interface FunctionFormFieldProps<T extends FieldValues> {
  control: Control<T>;
  name: FieldPath<T>;
  hideDefaultFunction?: boolean;
  label?: string;
  onSelect?: (value: string) => void;
}

export function FunctionFormField<T extends FieldValues>({
  control,
  name,
  hideDefaultFunction = false,
  label = "Function",
  onSelect,
}: FunctionFormFieldProps<T>) {
  const functions = useAllFunctionConfigs();

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <div className="grid gap-x-8 md:grid-cols-2">
            <FunctionSelector
              selected={field.value}
              onSelect={(value) => {
                field.onChange(value);
                onSelect?.(value);
              }}
              functions={functions}
              hideDefaultFunction={hideDefaultFunction}
            />
          </div>
        </FormItem>
      )}
    />
  );
}
