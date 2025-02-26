import { useForm, useWatch } from "react-hook-form";
import { useFetcher } from "react-router";
import { useEffect } from "react";
import { v7 as uuid } from "uuid";
import { type SFTFormValues, SFTFormValuesResolver } from "./types";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { MetricSelector } from "~/components/metric/MetricSelector";
import { VariantSelector } from "./VariantSelector";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { Button } from "~/components/ui/button";
import { Form } from "~/components/ui/form";
import type { ChatCompletionConfig } from "~/utils/config/variant";
import type { Config } from "~/utils/config";
import { models } from "./model_options";
import { useCountFetcher } from "~/routes/api/curated_inferences/count.route";

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

  // Separate fetcher for form submission
  const formFetcher = useFetcher();

  const watchedFields = useWatch({
    control: form.control,
    name: ["function", "metric", "threshold"] as const,
  });

  const [functionName, metricName, threshold] = watchedFields;
  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: threshold ?? undefined,
  });

  // Use formFetcher for submission errors
  const errorsOnSubmit = formFetcher.data?.errors;
  useEffect(() => {
    if (errorsOnSubmit) {
      setSubmissionPhase("idle");
    }
  }, [errorsOnSubmit, setSubmissionPhase]);

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
    <div className="mt-4">
      <Form {...form}>
        <form
          onSubmit={(e) => {
            handleSubmit(onSubmit)(e);
          }}
          className="space-y-6"
        >
          <div className="space-y-6">
            <div className="flex flex-col gap-1">
              <FunctionSelector<SFTFormValues>
                control={form.control}
                name="function"
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
              <MetricSelector<SFTFormValues>
                control={form.control}
                name="metric"
                functionFieldName="function"
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
