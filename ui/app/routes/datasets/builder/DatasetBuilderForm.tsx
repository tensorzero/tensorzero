import { useForm } from "react-hook-form";
import { Form } from "~/components/ui/form";
import { DatasetSelector } from "./DatasetSelector";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";

export function DatasetBuilderForm({
  dataset_counts,
}: {
  dataset_counts: DatasetCountInfo[];
}) {
  const form = useForm<DatasetBuilderFormValues>({
    defaultValues: {
      dataset_name: "",
      type: "chat",
      function_name: undefined,
      variant_name: undefined,
      metric_name: undefined,
      join_demonstrations: false,
    },
    resolver: DatasetBuilderFormValuesResolver,
    mode: "onChange",
  });

  return (
    <Form {...form}>
      <form className="space-y-6">
        <div className="space-y-6">
          <DatasetSelector
            control={form.control}
            dataset_counts={dataset_counts}
          />
        </div>
      </form>
    </Form>
  );
}
