import { useForm, useWatch } from "react-hook-form";
import { useFetcher } from "react-router";
import { useEffect } from "react";
import { v7 as uuid } from "uuid";
import { type SFTFormValues, SFTFormValuesResolver } from "./types";
import { FunctionFormField } from "~/components/function/FunctionFormField";
import CurationMetricSelector from "~/components/metric/CurationMetricSelector";
import { VariantSelector } from "./VariantSelector";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { Button } from "~/components/ui/button";
import { Form } from "~/components/ui/form";
import type {
  ChatCompletionConfig,
  UiConfig,
  VariantInfo,
} from "~/types/tensorzero";
import { models } from "./model_options";
import { useCountFetcher } from "~/routes/api/curated_inferences/count.route";
import { logger } from "~/utils/logger";
import { useFunctionConfig } from "~/context/config";
import { useReadOnly } from "~/context/read-only";

export function SFTForm({
  config,
  submissionPhase,
  setSubmissionPhase,
}: {
  config: UiConfig;
  submissionPhase: "idle" | "submitting" | "pending" | "complete";
  setSubmissionPhase: (
    phase: "idle" | "submitting" | "pending" | "complete",
  ) => void;
}) {
  const isReadOnly = useReadOnly();
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
  const functionConfig = useFunctionConfig(functionName);
  const parsedThreshold =
    typeof threshold === "string" ? parseFloat(threshold) : threshold;

  const counts = useCountFetcher({
    functionName: functionName ?? undefined,
    metricName: metricName ?? undefined,
    threshold: !isNaN(parsedThreshold) ? parsedThreshold : undefined,
  });
  const isCuratedInferenceCountLow =
    counts.curatedInferenceCount !== null && counts.curatedInferenceCount < 10;

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
      const cleanedThreshold =
        typeof data.threshold === "string"
          ? parseFloat(data.threshold)
          : data.threshold;

      if (isNaN(cleanedThreshold)) {
        logger.error("Threshold is not a valid number:", data.threshold);
        return;
      }

      const cleanedData = {
        ...data,
        threshold: cleanedThreshold,
      };

      const submitData = new FormData();
      submitData.append("data", JSON.stringify(cleanedData));

      formFetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      logger.error("Submission error (likely a bug):", error);
    }
  };

  const getChatCompletionVariantsForFunction = (): Record<
    string,
    ChatCompletionConfig
  > => {
    if (!functionConfig) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(functionConfig.variants || {})
        .filter(
          (entry): entry is [string, VariantInfo] =>
            entry[1]?.inner.type === "chat_completion",
        )
        .map(([name, variant]) => [
          name,
          variant.inner as ChatCompletionConfig,
        ]),
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
      case "idle":
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
              <FunctionFormField
                control={form.control}
                name="function"
                hideDefaultFunction={true}
              />

              {errors.function && (
                <p className="text-xs text-red-500">
                  {errors.function.message}
                </p>
              )}
            </div>

            <div className="flex flex-col">
              <CurationMetricSelector<SFTFormValues>
                control={form.control}
                name="metric"
                functionFieldName="function"
                config={config}
                addDemonstrations={true}
                feedbackCount={counts.feedbackCount}
                curatedInferenceCount={counts.curatedInferenceCount}
                isLoading={counts.isLoading}
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

          <Button
            type="submit"
            disabled={
              submissionPhase !== "idle" ||
              isCuratedInferenceCountLow ||
              isReadOnly
            }
          >
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
