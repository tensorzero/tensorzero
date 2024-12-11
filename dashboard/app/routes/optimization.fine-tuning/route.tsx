import {
  json,
  type LoaderFunctionArgs,
  type MetaFunction,
} from "@remix-run/node";
import { Form } from "~/components/ui/form";
import { Button } from "~/components/ui/button";
import { ModelOption, models } from "./mock-data";
import { useForm } from "react-hook-form";
// import { functions, metrics, models, promptTemplates } from "./mock-data";
import { Textarea } from "~/components/ui/textarea";
import { useEffect, useMemo, useState } from "react";
import { useConfig } from "~/context/config";
import {
  countFeedbacksForMetric,
  countInferencesForFunction,
  countCuratedInferences,
} from "~/utils/clickhouse";
import { getConfig } from "~/utils/config.server";
import { useLoaderData, useSearchParams } from "@remix-run/react";
import { ChatCompletionConfig } from "~/utils/config/variant";

import { FunctionSelector } from "./FunctionSelector";
import { MetricSelector } from "./MetricSelector";
import { VariantSelector } from "./VariantSelector";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { get_fine_tuned_model_config } from "~/utils/fine_tuning/openai";
export const meta: MetaFunction = () => {
  return [
    { title: "TensorZeroFine-Tuning Dashboard" },
    { name: "description", content: "Fine Tuning Optimization Dashboard" },
  ];
};
import { stringify } from "smol-toml";

export type FormValues = {
  function: string;
  metric: string;
  model: ModelOption;
  variant: string;
  validationSplit: number;
  maxSamples: number;
  threshold?: number;
};

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function");
  const metricName = url.searchParams.get("metric");

  let inferenceCount = null;
  let feedbackCount = null;
  let curatedInferenceCount = null;
  const config = await getConfig();
  if (functionName) {
    inferenceCount = await countInferencesForFunction(
      functionName,
      config.functions[functionName],
    );
  }
  if (metricName) {
    feedbackCount = await countFeedbacksForMetric(
      metricName,
      config.metrics[metricName],
    );
  }
  // TODO: count the curated inferences but don't actually return them
  if (functionName && metricName) {
    curatedInferenceCount = await countCuratedInferences(
      functionName,
      config.functions[functionName],
      metricName,
      config.metrics[metricName],
    );
  }
  return json({ inferenceCount, feedbackCount, curatedInferenceCount });
}

export default function FineTuning() {
  const { inferenceCount, feedbackCount, curatedInferenceCount } =
    useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();

  const config = useConfig();
  const form = useForm<FormValues>({
    defaultValues: {
      function: searchParams.get("function") || "",
      metric: searchParams.get("metric") || "",
      validationSplit: 20,
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

  const handleFunctionChange = (value: string) => {
    setSearchParams(
      (prev) => {
        if (value) {
          prev.set("function", value);
        } else {
          prev.delete("function");
        }
        return prev;
      },
      { replace: true },
    );
  };

  const handleMetricChange = (value: string) => {
    setSearchParams(
      (prev) => {
        prev.set("metric", value);
        return prev;
      },
      { replace: true },
    );
  };

  const getChatCompletionVariantsForFunction = useMemo((): Record<
    string,
    ChatCompletionConfig
  > => {
    const selectedFunction = searchParams.get("function");

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
  }, [config, searchParams]);

  useEffect(() => {
    if (inferenceCount !== null) {
      form.setValue("maxSamples", Math.min(100000, inferenceCount));
    }
  }, [inferenceCount, form]);

  async function onSubmit(data: FormValues) {
    try {
      setIsSubmitted(true);
      setSubmissionPhase("submitting");
      setSubmissionResult("Preparing training data...");
      console.log("data", data);
      console.log("data.model", data.model);

      // Call the API route for fine-tuning
      const response = await fetch("/api/fine-tuning", {
        method: "POST",
        body: JSON.stringify(data),
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || "Failed to start fine-tuning job");
      }

      const result = await response.json();
      console.log(result);
      setSubmissionResult("Job started. Polling for job status...");

      const job_id = result.job_id;
      let finished = false;
      let jobStatus = "";

      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 10000));
        const jobResponse = await fetch(`/api/fine-tuning/${job_id}`);
        const jobResult = await jobResponse.json();
        jobStatus = jobResult.status;
        setSubmissionResult(`Current job status: ${jobStatus}`);

        finished =
          jobStatus === "succeeded" ||
          jobStatus === "failed" ||
          jobStatus === "cancelled";
      }
      setSubmissionPhase("complete");

      const modelConfig = await get_fine_tuned_model_config(data.model.name);
      const fullModelConfig = { models: { [data.model.name]: modelConfig } };
      const oldVariantConfig =
        getChatCompletionVariantsForFunction[data.variant];
      const newVariantConfig = {
        ...oldVariantConfig,
        weight: 0,
        model_name: data.model.name,
      };
      const fullNewVariantConfig = {
        functions: {
          [data.function]: {
            variants: {
              [data.model.name]: newVariantConfig,
            },
          },
        },
      };
      const modelConfigToml = stringify(fullModelConfig);
      const newVariantConfigToml = stringify(fullNewVariantConfig);

      setFinalResult(
        `Model Configuration:\n\n${modelConfigToml}\n\n` +
          `New Variant Configuration:\n\n${newVariantConfigToml}`,
      );
    } catch (err) {
      const error = err as Error;
      setSubmissionPhase("complete");
      setFinalResult(`Error during fine-tuning: ${error.message}`);
    }

    //   setSubmissionResult(
    //     `Training data uploaded (File ID: ${file_id})\nStarting fine-tuning job...`
    //   );
    //   const job_id = await create_fine_tuning_job(data.model, file_id);

    //   setSubmissionPhase("pending");
    //   let finished = false;
    //   let job: OpenAI.FineTuning.FineTuningJob | undefined;
    //   let counter = 1;

    //   while (!finished) {
    //     await new Promise((resolve) => setTimeout(resolve, 10000));
    //     job = await poll_fine_tuning_job(job_id);

    //     // Update UI with current status
    //     counter++;
    //     setCounter(counter);
    //     setSubmissionResult(
    //       `Attempt ${counter}\n\nFine-tuning job status: ${job.status}\n` +
    //         `Training progress: ${job.trained_tokens ?? 0} tokens\n` +
    //         `${job.status === "running" ? "Training in progress..." : ""}`
    //     );

    //     finished =
    //       job.status === "succeeded" ||
    //       job.status === "failed" ||
    //       job.status === "cancelled";
    //   }
    //   if (!job) {
    //     throw new Error("No job found after fine-tuning");
    //   }

    //   setSubmissionPhase("complete");
    //   setFinalResult(
    //     job.status === "succeeded"
    //       ? `Fine-tuning completed successfully!\n\n` +
    //           `Model ID: ${job.fine_tuned_model}\n` +
    //           `Training tokens: ${job.trained_tokens}\n` +
    //           `Training file: ${job.training_file}\n` +
    //           `Validation file: ${job.validation_file ?? "None"}`
    //       : `Fine-tuning failed with status: ${job.status}\n` +
    //           `${job.error?.message ?? "No error message provided"}`
    //   );
    // } catch (err) {
    //   const error = err as Error;
    //   setSubmissionPhase("complete");
    //   setFinalResult(`Error during fine-tuning: ${error.message}`);
    // }
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
                  inferenceCount={inferenceCount}
                  config={config}
                  onFunctionChange={handleFunctionChange}
                />

                <MetricSelector
                  control={form.control}
                  feedbackCount={feedbackCount}
                  curatedInferenceCount={curatedInferenceCount}
                  config={config}
                  onMetricChange={handleMetricChange}
                />

                <VariantSelector
                  control={form.control}
                  chatCompletionVariants={getChatCompletionVariantsForFunction}
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
