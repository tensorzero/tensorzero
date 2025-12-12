import { useForm, useWatch } from "react-hook-form";
import { Form } from "~/components/ui/form";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import { FunctionFormField } from "~/components/function/FunctionFormField";
import { DatasetFormField } from "~/components/dataset/DatasetFormField";
import { useConfig, useFunctionConfig } from "~/context/config";
import CurationMetricSelector from "~/components/metric/CurationMetricSelector";
import { useCountFetcher } from "~/routes/api/curated_inferences/count.route";
import { useFetcher } from "react-router";
import { useEffect, useMemo, useState } from "react";
import { Button } from "~/components/ui/button";
import OutputSourceSelector from "./OutputSourceSelector";
import { DatasetCountDisplay } from "./DatasetCountDisplay";
import { logger } from "~/utils/logger";

export function DatasetBuilderForm() {
  const config = useConfig();
  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "complete"
  >("idle");
  const [countToInsert, setCountToInsert] = useState<number | null>(null);
  const [isNewDataset, setIsNewDataset] = useState<boolean | null>(null);
  // Track loading flags from child fetchers
  const [isMetricSelectorLoading, setIsMetricSelectorLoading] = useState(false);
  const [isOutputSourceLoading, setIsOutputSourceLoading] = useState(false);
  const [isInsertCountLoading, setIsInsertCountLoading] = useState(false);

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
    name: ["function", "metric_name", "threshold", "dataset"] as const,
  });

  const [functionName, metricName, threshold, selectedDataset] = watchedFields;
  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: threshold ?? undefined,
  });
  const functionConfig = useFunctionConfig(functionName ?? "");

  useEffect(() => {
    const metricConfig = config.metrics[metricName ?? ""];
    form.setValue("metric_config", metricConfig ? metricConfig : undefined);
    const functionType = functionConfig?.type;
    if (functionType) {
      form.setValue("type", functionType);
    }
  }, [metricName, functionName, config, form, functionConfig]);

  // Compute whether any part of the form is loading
  const isAnyLoading = useMemo(
    () =>
      counts.isLoading ||
      isMetricSelectorLoading ||
      isOutputSourceLoading ||
      isInsertCountLoading,
    [
      counts.isLoading,
      isMetricSelectorLoading,
      isOutputSourceLoading,
      isInsertCountLoading,
    ],
  );

  // Handle form submission response
  useEffect(() => {
    if (formFetcher.data) {
      if (formFetcher.data.errors) {
        logger.error("Form submission error:", formFetcher.data.errors);
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
      if (isAnyLoading) {
        return;
      }
      const submitData = new FormData();
      submitData.append("data", JSON.stringify(data));

      formFetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      logger.error("Submission error:", error);
      setSubmissionPhase("idle");
    }
  };

  function getButtonText(isNewDataset: boolean | null) {
    switch (submissionPhase) {
      case "submitting":
        return "Creating Dataset...";
      case "complete":
        return "Success";
      case "idle":
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
          <DatasetFormField
            control={form.control}
            name="dataset"
            label="Dataset"
            placeholder="Select a dataset"
            onSelect={(dataset, isNew) => {
              setIsNewDataset(isNew);
            }}
          />

          <FunctionFormField
            control={form.control}
            name="function"
            onSelect={() => {
              form.resetField("variant");
            }}
          />

          <CurationMetricSelector<DatasetBuilderFormValues>
            control={form.control}
            name="metric_name"
            functionFieldName="function"
            config={config}
            addDemonstrations={false}
            feedbackCount={counts.feedbackCount}
            curatedInferenceCount={counts.curatedInferenceCount}
            isLoading={counts.isLoading}
            onMetricsLoadingChange={setIsMetricSelectorLoading}
          />
          <OutputSourceSelector
            control={form.control}
            onLoadingChange={setIsOutputSourceLoading}
          />
        </div>
        <DatasetCountDisplay
          control={form.control}
          setCountToInsert={setCountToInsert}
          onLoadingChange={setIsInsertCountLoading}
        />
        <Button
          type="submit"
          disabled={
            submissionPhase !== "idle" ||
            isAnyLoading ||
            countToInsert === null ||
            countToInsert === 0 ||
            !selectedDataset
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
