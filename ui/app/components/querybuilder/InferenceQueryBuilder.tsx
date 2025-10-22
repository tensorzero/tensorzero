import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { forwardRef, useImperativeHandle } from "react";
import { Form } from "~/components/ui/form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs } from "~/context/config";
import type { Control } from "react-hook-form";
import type { InferenceFilter } from "tensorzero-node";
import { FieldValidation } from "./FilterRows";
import InferenceFilterBuilder from "./InferenceFilterBuilder";

// Validate current filter values (used on blur and submit)
function validate(filter: InferenceFilter | undefined): boolean {
  if (!filter) return true;

  if (filter.type === "tag") {
    const keyValid = FieldValidation.tagKey.safeParse(filter.key).success;
    const valueValid = FieldValidation.tagValue.safeParse(filter.value).success;
    return keyValid && valueValid;
  }

  if (filter.type === "float_metric") {
    return FieldValidation.floatValue.safeParse(filter.value.toString())
      .success;
  }

  if (filter.type === "boolean_metric") {
    return true; // Always valid (dropdown)
  }

  if (filter.type === "and" || filter.type === "or") {
    return filter.children.every((child) => validate(child));
  }

  return false; // Unknown filter type
}

const InferenceQueryBuilderSchema = z.object({
  function: z.string(),
});

export type InferenceQueryBuilderFormValues = z.infer<
  typeof InferenceQueryBuilderSchema
>;

export interface InferenceQueryBuilderRef {
  triggerValidation: () => Promise<boolean>;
}

interface InferenceQueryBuilderProps {
  inferenceFilter?: InferenceFilter;
  setInferenceFilter: React.Dispatch<
    React.SetStateAction<InferenceFilter | undefined>
  >;
}

interface FunctionFormFieldProps {
  control: Control<InferenceQueryBuilderFormValues>;
}

function FunctionFormField({ control }: FunctionFormFieldProps) {
  const functions = useAllFunctionConfigs();

  return (
    <FormField
      control={control}
      name="function"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Function</FormLabel>
          <FunctionSelector
            selected={field.value}
            onSelect={(value) => {
              field.onChange(value);
            }}
            functions={functions}
          />
        </FormItem>
      )}
    />
  );
}

export const InferenceQueryBuilder = forwardRef<
  InferenceQueryBuilderRef,
  InferenceQueryBuilderProps
>(function InferenceQueryBuilder(
  { inferenceFilter, setInferenceFilter }: InferenceQueryBuilderProps,
  ref,
) {
  const form = useForm<InferenceQueryBuilderFormValues>({
    defaultValues: {
      function: "",
    },
    resolver: zodResolver(InferenceQueryBuilderSchema),
    mode: "onChange",
  });

  // Expose validation trigger method to parent via ref
  useImperativeHandle(
    ref,
    () => ({
      triggerValidation: async () => {
        const formValid = await form.trigger();
        const filterValid = validate(inferenceFilter);
        return formValid && filterValid;
      },
    }),
    [inferenceFilter, form],
  );

  return (
    <Form {...form}>
      <form className="space-y-6">
        <FunctionFormField control={form.control} />
        <InferenceFilterBuilder
          inferenceFilter={inferenceFilter}
          setInferenceFilter={setInferenceFilter}
        />
      </form>
    </Form>
  );
});
