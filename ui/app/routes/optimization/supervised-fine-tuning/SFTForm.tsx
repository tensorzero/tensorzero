import { useForm, useWatch } from "react-hook-form";
import { useFetcher } from "react-router";
import { useEffect, useState } from "react";
import { v7 as uuid } from "uuid";
import { type SFTFormValues, SFTFormValuesResolver } from "./types";
import { FunctionSelector } from "./FunctionSelector";
import { MetricSelector } from "./MetricSelector";
import { VariantSelector } from "./VariantSelector";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { Button } from "~/components/ui/button";
import { Form } from "~/components/ui/form";
import type { ChatCompletionConfig } from "~/utils/config/variant";
import type { Config } from "~/utils/config";
import type { CountsData } from "~/routes/api/curated_inferences/count.route";
import { models } from "./model_options";

export function SFTForm({
  config,
  submissionPhase,
  setSubmissionPhase,
}: {
  config: Config;
  submissionPhase: "idle" | "submitting" | "pending" | "complete";
  setSubmissionPhase: (
    phase: "idle" | "submitting" | "pending" | "complete",
  ) => void;
}) {
  const form = useForm<SFTFormValues>({
    defaultValues: {
      function: "",
      metric: "",
      validationSplitPercent: 20,
      maxSamples: 100000,
      threshold: 0.5,
      jobId: uuid(),
    },
    resolver: SFTFormValuesResolver,
    mode: "onChange",
  });

  const {
    handleSubmit,
    formState: { errors },
  } = form;

  // Separate fetchers for counts and form submission
  const countFetcher = useFetcher();
  const formFetcher = useFetcher();

  const watchedValues = useWatch({
    control: form.control,
    name: ["function", "metric", "threshold"],
  });

  // Use formFetcher for submission errors
  const errorsOnSubmit = formFetcher.data?.errors;
  useEffect(() => {
    if (errorsOnSubmit) {
      setSubmissionPhase("idle");
    }
  }, [errorsOnSubmit, setSubmissionPhase]);

  const [counts, setCounts] = useState<CountsData>({
    inferenceCount: null,
    feedbackCount: null,
    curatedInferenceCount: null,
  });

  // Update counts only from countFetcher
  useEffect(() => {
    if (countFetcher.data && !countFetcher.data.errors) {
      setCounts(countFetcher.data as CountsData);
    }
  }, [countFetcher.data]);

  // Handle count fetching with countFetcher
  useEffect(() => {
    const [functionName, metricName, threshold] = watchedValues;

    if (functionName) {
      const params = new URLSearchParams();
      params.set("function", functionName);
      if (metricName) params.set("metric", metricName);
      if (threshold) params.set("threshold", String(threshold));

      countFetcher.load(`/api/curated_inferences/count?${params}`);
    }
  }, [watchedValues]);

  // Form submission using formFetcher
  const onSubmit = async (data: SFTFormValues) => {
    try {
      const submitData = new FormData();
      submitData.append("data", JSON.stringify(data));

      formFetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      console.error("Submission error (likely a bug):", error);
    }
  };

  const getChatCompletionVariantsForFunction = (): Record<
    string,
    ChatCompletionConfig
  > => {
    const selectedFunction = form.getValues("function");

    if (!selectedFunction || !config?.functions[selectedFunction]) {
      return {};
    }

    const functionConfig = config.functions[selectedFunction];
    return Object.fromEntries(
      Object.entries(functionConfig.variants || {}).filter(
        (entry): entry is [string, ChatCompletionConfig] =>
          entry[1].type === "chat_completion",
      ),
    );
  };

  // Sets the max samples limit based on the number of curatedInferences (if available) or inferences (if available)
  // This means it will change when the function is selected or the metric is changed to something that actually curates inferences (i.e. not None)
  useEffect(() => {
    if (counts.curatedInferenceCount !== null) {
      form.setValue(
        "maxSamples",
        Math.min(100000, counts.curatedInferenceCount),
      );
    } else if (counts.inferenceCount !== null) {
      form.setValue("maxSamples", Math.min(100000, counts.inferenceCount));
    }
  }, [counts.curatedInferenceCount, counts.inferenceCount, form]);

  function getButtonText() {
    switch (submissionPhase) {
      case "submitting":
        return "Submitting...";
      case "pending":
        return "Pending...";
      case "complete":
        return "Complete";
      default:
        return "Start Fine-tuning Job";
    }
  }

  return (
    <div className="mt-8">
      <Form {...form}>
        <form
          onSubmit={(e) => {
            handleSubmit(onSubmit)(e);
          }}
          className="space-y-6"
        >
          <div className="space-y-6">
            <div className="flex flex-col gap-1">
              <FunctionSelector
                control={form.control}
                inferenceCount={counts.inferenceCount}
                config={config}
              />
              {errors.function && (
                <p className="text-xs text-red-500">
                  {errors.function.message}
                </p>
              )}
            </div>

            <div className="flex flex-col">
              <MetricSelector
                control={form.control}
                feedbackCount={counts.feedbackCount}
                curatedInferenceCount={counts.curatedInferenceCount}
                config={config}
              />

              {errors.metric && (
                <p className="text-xs text-red-500">{errors.metric.message}</p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <VariantSelector
                control={form.control}
                chatCompletionVariants={getChatCompletionVariantsForFunction()}
              />

              {errors.variant && (
                <p className="text-xs text-red-500">{errors.variant.message}</p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <ModelSelector control={form.control} models={models} />
              {errors.model && (
                <p className="text-xs text-red-500">{errors.model.message}</p>
              )}
            </div>
            <AdvancedParametersAccordion
              control={form.control}
              maxSamplesLimit={counts.inferenceCount ?? undefined}
            />
          </div>

          <Button type="submit" disabled={submissionPhase !== "idle"}>
            {getButtonText()}
          </Button>
          {errorsOnSubmit && (
            <p className="text-sm text-red-500">{errorsOnSubmit.message}</p>
          )}
        </form>
      </Form>
    </div>
  );
}
