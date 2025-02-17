import { useForm } from "react-hook-form";
import { Form } from "~/components/ui/form";
import { DatasetSelector } from "./DatasetSelector";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useConfig } from "~/context/config";
import { useEffect, useState } from "react";
import { useFetcher } from "react-router";

export function DatasetBuilderForm({
  dataset_counts,
}: {
  dataset_counts: DatasetCountInfo[];
}) {
  const config = useConfig();
  const countFetcher = useFetcher();
  const [inferenceCount, setInferenceCount] = useState<number | null>(null);

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

  // Watch for function name changes to update inference count
  const functionName = form.watch("function_name");
  useEffect(() => {
    if (functionName) {
      const params = new URLSearchParams();
      params.set("function", functionName);
      countFetcher.load(`/api/curated_inferences/count?${params}`);
    } else {
      setInferenceCount(null);
    }
  }, [functionName]);

  // Update inference count when data is loaded
  useEffect(() => {
    if (countFetcher.data) {
      setInferenceCount(countFetcher.data.inferenceCount);
    }
  }, [countFetcher.data]);

  return (
    <Form {...form}>
      <form className="space-y-6">
        <div className="space-y-6">
          <DatasetSelector
            control={form.control}
            dataset_counts={dataset_counts}
          />
          <FunctionSelector<DatasetBuilderFormValues>
            control={form.control}
            name="function_name"
            inferenceCount={inferenceCount}
            config={config}
          />
        </div>
      </form>
    </Form>
  );
}
