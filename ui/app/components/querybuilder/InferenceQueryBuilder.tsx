import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form } from "~/components/ui/form";
import type { InferenceFilter } from "tensorzero-node";
import InferenceFilterBuilder, {
  validateInferenceFilter,
} from "./InferenceFilterBuilder";
import FunctionFormField from "./FunctionFormField";

const InferenceQueryBuilderSchema = z
  .object({
    function: z.string(),
    inferenceFilter: z.custom<InferenceFilter | undefined>().optional(),
  })
  .refine(
    (data) => validateInferenceFilter(data.inferenceFilter),
    "Invalid filter configuration",
  );

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

  return (
    <Form {...form}>
      <form className="space-y-6" onSubmit={form.handleSubmit(handleSubmit)}>
        <FunctionFormField control={form.control} />
        <InferenceFilterBuilder
          inferenceFilter={form.watch("inferenceFilter")}
          setInferenceFilter={(filter) =>
            form.setValue("inferenceFilter", filter, {
              shouldValidate: true,
              shouldDirty: true,
            })
          }
        />
      </form>
    </Form>
  );
}
