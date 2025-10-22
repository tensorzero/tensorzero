import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form } from "~/components/ui/form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs } from "~/context/config";
import type { Control } from "react-hook-form";
import type { InferenceFilter } from "tensorzero-node";
import InferenceFilterBuilder from "./InferenceFilterBuilder";

const InferenceQueryBuilderSchema = z.object({
  function: z.string(),
});

export type InferenceQueryBuilderFormValues = z.infer<
  typeof InferenceQueryBuilderSchema
>;

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

export default function InferenceQueryBuilder({
  inferenceFilter,
  setInferenceFilter,
}: InferenceQueryBuilderProps) {
  const form = useForm<InferenceQueryBuilderFormValues>({
    defaultValues: {
      function: "",
    },
    resolver: zodResolver(InferenceQueryBuilderSchema),
    mode: "onChange",
  });

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
}
