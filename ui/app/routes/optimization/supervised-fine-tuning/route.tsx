import { useFetcher, type MetaFunction } from "react-router";
import { useEffect, useState } from "react";
import {
  type SFTFormValues,
  SFTFormValuesSchema,
  SFTFormValuesResolver,
} from "./types";
import { v7 as uuid } from "uuid";
import type { SFTJob } from "~/utils/supervised_fine_tuning/common";
import { models } from "./model_options";
import { useRevalidator } from "react-router";
import { useForm } from "react-hook-form";
import { redirect } from "react-router";
import { launch_sft_job } from "~/utils/supervised_fine_tuning/client";
import type { ChatCompletionConfig } from "~/utils/config/variant";
import { useConfig } from "~/context/config";
import { FunctionSelector } from "./FunctionSelector";
import { MetricSelector } from "./MetricSelector";
import { VariantSelector } from "./VariantSelector";
import {
  dump_model_config,
  get_fine_tuned_model_config,
} from "~/utils/config/models";
import { ModelSelector } from "./ModelSelector";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { Form } from "~/components/ui/form";
import type { Route } from "./+types/route";
// The following import would be needed for type-safe fetching of counts
// import type { Route as CuratedInferencesCount } from "../../api/curated_inferences/+types/count.route";
import type { CountsData } from "../../api/curated_inferences/count.route";
import type { Config } from "~/utils/config";

export const meta: MetaFunction = () => {
  return [
    { title: "TensorZero Supervised Fine-Tuning UI" },
    {
      name: "description",
      content: "Supervised Fine-Tuning Optimization UI",
    },
  ];
};

// Mutable store mapping job IDs to their info
export const jobStore: { [jobId: string]: SFTJob } = {};

// If there is a job_id in the URL, grab it from the job store and pull it.
export async function loader({ params }: Route.LoaderArgs) {
  const job_id = params.job_id;

  if (!job_id) {
    return { status: "idle" };
  }

  const storedJob = jobStore[job_id];
  if (!storedJob) {
    throw new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
    });
  }
  if (storedJob.status() === "completed" || storedJob.status() === "error") {
    return {
      status: storedJob.status(),
      result: storedJob.result(),
      modelProvider: storedJob.provider(),
    };
  }

  try {
    // Poll for updates
    const updatedJob = await storedJob.poll();
    jobStore[job_id] = updatedJob;

    const result = updatedJob.result();
    const status = updatedJob.status();
    const modelProvider = updatedJob.provider();

    return {
      status,
      result,
      modelProvider,
    };
  } catch (error) {
    return {
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

// The action actually launches the fine-tuning job.
export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const serializedFormData = formData.get("data");
  if (!serializedFormData || typeof serializedFormData !== "string") {
    throw new Error("Form data must be provided");
  }

  const jsonData = JSON.parse(serializedFormData);
  const validatedData = SFTFormValuesSchema.parse(jsonData);

  const job = await launch_sft_job(validatedData);
  jobStore[validatedData.jobId] = job;

  return redirect(
    `/optimization/supervised-fine-tuning/${validatedData.jobId}`,
  );
}

// Renders the fine-tuning form and status info.
export default function SupervisedFineTuning({
  loaderData,
}: Route.ComponentProps) {
  const config = useConfig();
  const { status, result, modelProvider } = loaderData;
  const revalidator = useRevalidator();

  // If running, periodically poll for updates on the job
  useEffect(() => {
    if (status === "running") {
      setSubmissionPhase("pending");
      const interval = setInterval(() => {
        revalidator.revalidate();
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [status, revalidator]);

  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");
  const finalResult = result
    ? dump_model_config(get_fine_tuned_model_config(result, modelProvider))
    : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <main>
        <h2 className="mb-4 text-2xl font-semibold">Supervised Fine-Tuning</h2>
        <div className="mb-6 h-px w-full bg-gray-200"></div>
        {status === "idle" && (
          <FineTuningForm
            config={config}
            submissionPhase={submissionPhase}
            setSubmissionPhase={setSubmissionPhase}
          />
        )}

        {status !== "idle" && (
          <div className="mt-4 rounded-lg bg-gray-100 p-4">
            <div className="mb-2 font-medium">
              Loader Data (Last Updated: {new Date().toLocaleTimeString()})
            </div>
            <Textarea
              value={JSON.stringify(loaderData, null, 2)}
              className="h-48 w-full resize-none border-none bg-transparent focus:ring-0"
              readOnly
            />
          </div>
        )}

        {finalResult && (
          <div className="mt-4 rounded-lg bg-gray-100 p-4">
            <div className="mb-2 font-medium">Configuration</div>
            <Textarea
              value={finalResult}
              className="h-48 w-full resize-none border-none bg-transparent focus:ring-0"
              readOnly
            />
          </div>
        )}
      </main>
    </div>
  );
}

function FineTuningForm({
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

  const fetcher = useFetcher();

  const [counts, setCounts] = useState<CountsData>({
    inferenceCount: null,
    feedbackCount: null,
    curatedInferenceCount: null,
  });

  // Add this effect to watch for fetcher data changes
  // This is needed because fetcher.load is not async, so it doesn't return a promise
  // and we need to watch for data changes via the fetcher.data property
  useEffect(() => {
    if (fetcher.data) {
      setCounts(fetcher.data as CountsData);
    }
  }, [fetcher.data]);

  const fetchCounts = (
    functionName?: string,
    metricName?: string,
    threshold?: number,
  ) => {
    const params = new URLSearchParams();
    if (functionName) params.set("function", functionName);
    if (metricName) params.set("metric", metricName);
    if (threshold) params.set("threshold", String(threshold));

    fetcher.load(`/api/curated_inferences/count?${params}`);
  };

  const handleFunctionChange = (value: string) => {
    fetchCounts(
      value,
      form.getValues("metric") || undefined,
      form.getValues("threshold") || undefined,
    );
  };

  const handleMetricChange = (value: string | null) => {
    fetchCounts(
      form.getValues("function") || undefined,
      value || undefined,
      form.getValues("threshold") || undefined,
    );
  };

  const handleThresholdChange = (value: number) => {
    fetchCounts(
      form.getValues("function") || undefined,
      form.getValues("metric") || undefined,
      value,
    );
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

  const onSubmit = async (data: SFTFormValues) => {
    try {
      const formData = {
        ...data,
      };

      const submitData = new FormData();
      submitData.append("data", JSON.stringify(formData));

      fetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      console.error("Submission error:", error);
    }
  };

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
            <FunctionSelector
              control={form.control}
              inferenceCount={counts.inferenceCount}
              config={config}
              onFunctionChange={handleFunctionChange}
            />
            {errors.function && (
              <p className="text-sm text-red-500">{errors.function.message}</p>
            )}

            <MetricSelector
              control={form.control}
              feedbackCount={counts.feedbackCount}
              curatedInferenceCount={counts.curatedInferenceCount}
              config={config}
              onMetricChange={handleMetricChange}
              onThresholdChange={handleThresholdChange}
            />
            {errors.metric && (
              <p className="text-sm text-red-500">{errors.metric.message}</p>
            )}

            <VariantSelector
              control={form.control}
              chatCompletionVariants={getChatCompletionVariantsForFunction()}
            />

            <ModelSelector control={form.control} models={models} />

            <AdvancedParametersAccordion
              control={form.control}
              maxSamplesLimit={counts.inferenceCount ?? undefined}
            />
          </div>

          <Button type="submit" disabled={submissionPhase !== "idle"}>
            {getButtonText()}
          </Button>
        </form>
      </Form>
    </div>
  );
}
