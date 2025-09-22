import { useEffect } from "react";
import { useForm, useWatch } from "react-hook-form";
import { useFetcher } from "react-router";
import type {
  ChatCompletionConfig,
  Config,
  VariantInfo,
} from "tensorzero-node";
import { v7 as uuid } from "uuid";
import { FunctionFormField } from "~/components/function/FunctionFormField";
import { useCountData } from "~/components/metric/useCountData";
import { Button } from "~/components/ui/button";
import { Form } from "~/components/ui/form";
import { useAllFunctionConfigs, useFunctionConfig } from "~/context/config";
import { logger } from "~/utils/logger";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { FiltersInput } from "./Filters";
import { models } from "./model_options";
import { ModelSelector } from "./ModelSelector";
import { SFTFormValuesResolver, type SFTFormValues } from "./types";
import { VariantSelector } from "./VariantSelector";

// const metricTemplate = { metric: "", threshold: 0.5 };

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
      function: "generate_secret",
      metric: "",
      filters: [
        { metric: "goated", threshold: 0.5 },
        { metric: "elapsed_ms", threshold: 0.5 },
        { metric: "num_questions", threshold: 0.5 },
      ],
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

  const { counts, isCuratedInferenceCountLow } = useCountData({
    functionName,
    metricName,
    parsedThreshold,
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
                functions={useAllFunctionConfigs()}
                hideDefaultFunction={true}
              />

              {errors.function && (
                <p className="text-xs text-red-500">
                  {errors.function.message}
                </p>
              )}
            </div>

            <FiltersInput config={config} form={form} counts={counts} />

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
            disabled={submissionPhase !== "idle" || isCuratedInferenceCountLow}
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
