import { useForm, useController } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form } from "~/components/ui/form";
import InferenceFilterBuilder from "./InferenceFilterBuilder";
import FunctionFormField from "./FunctionFormField";
import { InferenceFilterSchema } from "./inference-filter-schema";

const InferenceQueryBuilderSchema = z.object({
  function: z.string().min(1, "Function is required"),
  inferenceFilter: InferenceFilterSchema.optional(),
});

export type InferenceQueryBuilderFormValues = z.infer<
  typeof InferenceQueryBuilderSchema
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
    resolver: zodResolver(InferenceQueryBuilderSchema),
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
