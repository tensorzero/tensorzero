import { type MetaFunction } from "react-router";
import { Form } from "~/components/ui/form";
import { Button } from "~/components/ui/button";
import { models } from "./model_options";
import { useForm } from "react-hook-form";
import { Textarea } from "~/components/ui/textarea";
import { useEffect, useState } from "react";
import { useConfig } from "~/context/config";
import {
  ChatCompletionConfig,
  create_dump_variant_config,
} from "~/utils/config/variant";

import { FunctionSelector } from "./FunctionSelector";
import { MetricSelector } from "./MetricSelector";
import { VariantSelector } from "./VariantSelector";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import {
  dump_model_config,
  get_fine_tuned_model_config,
} from "~/utils/fine_tuning/config_block";
import { launch_sft_job } from "~/utils/fine_tuning/client";
import { SFTFormValues } from "./types";

export const meta: MetaFunction = () => {
  return [
    { title: "TensorZeroFine-Tuning Dashboard" },
    { name: "description", content: "Fine Tuning Optimization Dashboard" },
  ];
};

export default function FineTuning() {
  const config = useConfig();
  const form = useForm<SFTFormValues>({
    defaultValues: {
      function: "",
      metric: "",
      validationSplitPercent: 20,
      maxSamples: 100000,
      threshold: 0.5,
    },
  });

  const [submissionResult, setSubmissionResult] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<string | null>(null);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");

  const [counts, setCounts] = useState<{
    inferenceCount: number | null;
    feedbackCount: number | null;
    curatedInferenceCount: number | null;
  }>({
    inferenceCount: null,
    feedbackCount: null,
    curatedInferenceCount: null,
  });

  const fetchCounts = async (functionName?: string, metricName?: string) => {
    const params = new URLSearchParams();
    if (functionName) params.set("function", functionName);
    if (metricName) params.set("metric", metricName);

    const response = await fetch(`/api/curated_inferences/count?${params}`);
    const data = await response.json();
    setCounts(data);
  };

  const handleFunctionChange = (value: string) => {
    fetchCounts(value, form.getValues("metric") || undefined);
  };

  const handleMetricChange = (value: string) => {
    fetchCounts(form.getValues("function") || undefined, value);
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

  useEffect(() => {
    if (counts.inferenceCount !== null) {
      form.setValue("maxSamples", Math.min(100000, counts.inferenceCount));
    }
  }, [counts.inferenceCount, form]);

  async function onSubmit(data: SFTFormValues) {
    try {
      setIsSubmitted(true);
      setSubmissionPhase("submitting");
      setSubmissionResult("Preparing training data...");

      let job = await launch_sft_job(data);
      setSubmissionResult("Job started. Polling for job status...");

      let finished = false;
      let result: string | undefined;

      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 10000));
        job = await job.poll();
        result = job.result();
        finished = result !== undefined;
        setSubmissionResult(job.display());
      }
      setSubmissionPhase("complete");
      const fineTunedModelName = job.result();
      if (!fineTunedModelName) {
        throw new Error("No fine-tuned model name found");
      }

      const modelConfig = await get_fine_tuned_model_config(
        fineTunedModelName,
        data.model.provider,
      );
      const modelConfigToml = dump_model_config(modelConfig);
      const oldVariantConfig =
        getChatCompletionVariantsForFunction()[data.variant];
      const newVariantConfigToml = create_dump_variant_config(
        oldVariantConfig,
        fineTunedModelName,
        data.function,
      );

      setFinalResult(
        `Model Configuration:\n\n${modelConfigToml}\n\n` +
          `New Variant Configuration:\n\n${newVariantConfigToml}`,
      );
    } catch (err) {
      const error = err as Error;
      setSubmissionPhase("complete");
      setFinalResult(`Error during fine-tuning: ${error.message}`);
    }
  }

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
    <div className="min-h-screen bg-gray-50">
      <main className="p-4">
        <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">
          Fine-Tuning
        </h2>

        <div className="mt-8">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <div className="space-y-6">
                <FunctionSelector
                  control={form.control}
                  inferenceCount={counts.inferenceCount}
                  config={config}
                  onFunctionChange={handleFunctionChange}
                />

                <MetricSelector
                  control={form.control}
                  feedbackCount={counts.feedbackCount}
                  curatedInferenceCount={counts.curatedInferenceCount}
                  config={config}
                  onMetricChange={handleMetricChange}
                />

                <VariantSelector
                  control={form.control}
                  chatCompletionVariants={getChatCompletionVariantsForFunction()}
                />

                <ModelSelector control={form.control} models={models} />

                <AdvancedParametersAccordion control={form.control} />
              </div>

              <div className="space-y-4">
                <Button
                  type="submit"
                  disabled={
                    !form.watch("function") ||
                    !form.watch("metric") ||
                    !form.watch("model") ||
                    !form.watch("variant") ||
                    form.formState.isSubmitting ||
                    isSubmitted
                  }
                >
                  {getButtonText()}
                </Button>

                {submissionResult && (
                  <div className="p-4 bg-gray-100 rounded-lg">
                    <div className="mb-2 font-medium">Job Status</div>
                    <Textarea
                      value={submissionResult}
                      className="w-full h-48 resize-none bg-transparent border-none focus:ring-0"
                      readOnly
                    />
                  </div>
                )}

                {finalResult && (
                  <div className="p-4 bg-gray-100 rounded-lg">
                    <div className="mb-2 font-medium">Configuration</div>
                    <Textarea
                      value={finalResult}
                      className="w-full h-48 resize-none bg-transparent border-none focus:ring-0"
                      readOnly
                    />
                  </div>
                )}
              </div>
            </form>
          </Form>
        </div>
      </main>
    </div>
  );
}
