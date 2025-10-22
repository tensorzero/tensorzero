import { useForm, useController } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form } from "~/components/ui/form";
import InferenceFilterBuilder from "./InferenceFilterBuilder";
import FunctionFormField from "./FunctionFormField";
import { inferenceFilterSchema } from "./inference-filter-schema";

const InferenceQueryBuilderFormValuesSchema = z.object({
  function: z.string().min(1, "Function is required"),
  inferenceFilter: inferenceFilterSchema.optional(),
});

export type InferenceQueryBuilderFormValues = z.infer<
  typeof InferenceQueryBuilderFormValuesSchema
>;

interface InferenceQueryBuilderProps {
  onSubmit?: (values: InferenceQueryBuilderFormValues) => void;
  defaultValues?: Partial<InferenceQueryBuilderFormValues>;
}

export default function InferenceQueryBuilder({
  onSubmit,
  defaultValues,
}: InferenceQueryBuilderProps) {
  const form = useForm<InferenceQueryBuilderFormValues>({
    defaultValues: {
      function: "",
      inferenceFilter: undefined,
      ...defaultValues,
    },
    resolver: zodResolver(InferenceQueryBuilderFormValuesSchema),
    mode: "onChange",
  });

  const handleSubmit = (values: InferenceQueryBuilderFormValues) => {
    onSubmit?.(values);
  };

  const {
    field: { value: inferenceFilter, onChange: setInferenceFilter },
  } = useController({
    name: "inferenceFilter",
    control: form.control,
  });

  return (
    <Form {...form}>
      <form className="space-y-6" onSubmit={form.handleSubmit(handleSubmit)}>
        <FunctionFormField control={form.control} />
        <InferenceFilterBuilder
          inferenceFilter={inferenceFilter}
          setInferenceFilter={setInferenceFilter}
        />
      </form>
    </Form>
  );
}
