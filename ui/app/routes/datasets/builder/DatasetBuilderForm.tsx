import { useForm, useWatch } from "react-hook-form";
import { Form } from "~/components/ui/form";
import { DatasetSelector } from "./DatasetSelector";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useConfig } from "~/context/config";
import { MetricSelector } from "~/components/metric/MetricSelector";
import { useCountFetcher } from "~/routes/api/curated_inferences/count.route";
import { useFetcher } from "react-router";
import { useEffect, useState } from "react";
import { Button } from "~/components/ui/button";
import OutputSourceSelector from "./OutputSourceSelector";
import { DatasetCountDisplay } from "./DatasetCountDisplay";

export function DatasetBuilderForm({
  dataset_counts,
}: {
  dataset_counts: DatasetCountInfo[];
}) {
  const config = useConfig();
  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "complete"
  >("idle");
  const [countToInsert, setCountToInsert] = useState<number | null>(null);
  const [isNewDataset, setIsNewDataset] = useState<boolean | null>(null);

  const form = useForm<DatasetBuilderFormValues>({
    defaultValues: {
      dataset: "",
      type: "chat",
      function: undefined,
      variant: undefined,
      metric_name: null,
      metric_config: undefined,
      threshold: 0.5,
      output_source: "none",
    },
    resolver: DatasetBuilderFormValuesResolver,
    mode: "onChange",
  });

  const { handleSubmit } = form;

  const formFetcher = useFetcher();

  const watchedFields = useWatch({
    control: form.control,
    name: ["function", "metric_name", "threshold"] as const,
  });

  const [functionName, metricName, threshold] = watchedFields;
  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: threshold ?? undefined,
  });
  useEffect(() => {
    const metricConfig = config.metrics[metricName ?? ""];
    form.setValue("metric_config", metricConfig ? metricConfig : undefined);
    const functionType = config.functions[functionName ?? ""]?.type;
    form.setValue("type", functionType);
  }, [metricName, functionName, config, form]);

  // Handle form submission response
  useEffect(() => {
    if (formFetcher.data) {
      if (formFetcher.data.errors) {
        console.error("Form submission error:", formFetcher.data.errors);
        setSubmissionPhase("idle");
        form.setError("root", {
          type: "submit",
          message:
            formFetcher.data.errors.message ||
            "An error occurred while processing your request",
        });
      } else if (formFetcher.data.success) {
        setSubmissionPhase("complete");
        form.clearErrors("root");
      }
    }
  }, [formFetcher.data, form]);

  // Form submission handler
  const onSubmit = async (data: DatasetBuilderFormValues) => {
    try {
      const submitData = new FormData();
      submitData.append("data", JSON.stringify(data));

      formFetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      console.error("Submission error:", error);
      setSubmissionPhase("idle");
    }
  };

  function getButtonText(isNewDataset: boolean | null) {
    switch (submissionPhase) {
      case "submitting":
        return "Creating Dataset...";
      case "complete":
        return "Success";
      default:
        if (isNewDataset) {
          return "Create Dataset";
        } else {
          return "Insert Into Dataset";
        }
    }
  }

  return (
    <Form {...form}>
      <form
        onSubmit={(e) => {
          handleSubmit(onSubmit)(e);
        }}
        className="space-y-6"
      >
        <div className="space-y-6">
          <DatasetSelector
            control={form.control}
            dataset_counts={dataset_counts}
            setIsNewDataset={setIsNewDataset}
          />
          <FunctionSelector<DatasetBuilderFormValues>
            control={form.control}
            name="function"
            inferenceCount={counts.inferenceCount}
            config={config}
          />
          <MetricSelector<DatasetBuilderFormValues>
            control={form.control}
            name="metric_name"
            functionFieldName="function"
            feedbackCount={counts.feedbackCount}
            curatedInferenceCount={counts.curatedInferenceCount}
            config={config}
            removeDemonstrations={true}
          />
          <OutputSourceSelector control={form.control} />
        </div>
        <DatasetCountDisplay
          control={form.control}
          setCountToInsert={setCountToInsert}
        />
        <Button
          type="submit"
          disabled={
            submissionPhase !== "idle" ||
            countToInsert === null ||
            countToInsert === 0
          }
          onClick={() => {
            if (submissionPhase === "complete") {
              setSubmissionPhase("idle");
              form.clearErrors("root");
            }
          }}
        >
          {getButtonText(isNewDataset)}
        </Button>
        {form.formState.errors.root && (
          <p className="mt-2 text-sm text-red-500">
            {form.formState.errors.root.message}
          </p>
        )}
      </form>
    </Form>
  );
}
