import type { Control, FieldPath, FieldValues } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";

interface DatasetFormFieldProps<T extends FieldValues> {
  control: Control<T>;
  name: FieldPath<T>;
  label?: string;
  onSelect?: (value: string, isNew: boolean) => void;
  placeholder: string;
  allowCreation?: boolean;
}

/**
 * This component should be used with react-hook-form only.
 * For standard RR7 forms we can just use the DatasetSelector component directly.
 */
export function DatasetFormField<T extends FieldValues>({
  control,
  name,
  label = "Dataset",
  onSelect,
  placeholder,
  allowCreation,
}: DatasetFormFieldProps<T>) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <div className="grid gap-x-8 md:grid-cols-2">
            <DatasetSelector
              selected={field.value}
              onSelect={(value, isNew) => {
                field.onChange(value);
                onSelect?.(value, isNew);
              }}
              placeholder={placeholder}
              allowCreation={allowCreation}
            />
          </div>
        </FormItem>
      )}
    />
  );
}
