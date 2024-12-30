import { data, useFetcher, type MetaFunction } from "react-router";
import { useEffect, useState } from "react";
import {
  type SFTFormValues,
  SFTFormValuesSchema,
  SFTFormValuesResolver,
} from "./types";
import { v7 as uuid } from "uuid";
import type {
  SFTJob,
  SFTJobStatus,
} from "~/utils/supervised_fine_tuning/common";
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
  type ProviderType,
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
import { ProgressIndicator, type ProgressInfo } from "./ProgressIndicator";

export const meta: MetaFunction = () => {
  return [
    { title: "TensorZero Supervised Fine-Tuning Dashboard" },
    {
      name: "description",
      content: "Supervised Fine-Tuning Optimization Dashboard",
    },
  ];
};

// Mutable store mapping job IDs to their info
export const jobStore: { [jobId: string]: SFTJob } = {};

// TODO: remove once we're happy
function get_progress_fixture(provider: ProviderType): LoaderData {
  switch (provider) {
    case "openai":
      return {
        status: "running",
        result: undefined,
        modelProvider: "openai",
        progressInfo: {
          jobUrl: "https://platform.openai.com/finetune/ftjob-abc123",
          provider: "openai",
          data: {
            object: "fine_tuning.job",
            id: "ftjob-abc123",
            model: "davinci-002",
            created_at: 1692661014,
            finished_at: 1692661190,
            fine_tuned_model: "ft:davinci-002:my-org:custom_suffix:7q8mpxmy",
            organization_id: "org-123",
            result_files: ["file-abc123"],
            status: "succeeded",
            validation_file: null,
            training_file: "file-abc123",
            hyperparameters: {
              n_epochs: 4,
              batch_size: 1,
              learning_rate_multiplier: 1.0,
            },
            trained_tokens: 5768,
            integrations: [],
            seed: 0,
            estimated_finish: 0,
            method: {
              type: "supervised",
              supervised: {
                hyperparameters: {
                  n_epochs: 4,
                  batch_size: 1,
                  learning_rate_multiplier: 1.0,
                },
              },
            },
          },
          estimatedCompletionTimestamp: Math.floor(Date.now()) + 300000,
        },
      };
    case "fireworks":
      return {
        status: "running",
        modelProvider: "fireworks",
        result: undefined,
        progressInfo: {
          provider: "fireworks",
          data: {
            state: "PENDING",
            modelId: "",
            baseModel: "accounts/fireworks/models/llama-v3p1-8b-instruct",
            batchSize: 16,
            createTime: "2024-12-28T15:16:43.649092Z",
            createdBy: "viraj@tensorzero.com",
            dataset:
              "accounts/viraj-ebfe5a/datasets/01940dd7-4f65-716a-a257-6da73412fe87",
            evaluationSplit: 0.2,
            evaluation: false,
            evaluationDataset: "",
            learningRate: 0.0001,
            loraRank: 8,
            loraTargetModules: [],
            maskToken: "",
            microBatchSize: 0,
            name: "accounts/viraj-ebfe5a/fineTuningJobs/ed8f3dead9d74b7fbe0623f039d151e3",
            padToken: "",
            status: {
              code: "OK",
              message: "",
            },
          },
          jobUrl: "https://fireworks.ai/dashboard/fine-tuning/ftjob-abc123",
        },
      };
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}
interface LoaderData {
  status: SFTJobStatus;
  result: string | undefined;
  modelProvider: ProviderType | undefined;
  progressInfo: ProgressInfo | undefined;
}

// If there is a job_id in the URL, grab it from the job store and pull it.
export async function loader({
  params,
}: Route.LoaderArgs): Promise<LoaderData | { status: "error"; error: string }> {
  // for debugging ProgressIndicator without starting a real job
  // return get_progress_fixture("fireworks");
  const job_id = params.job_id;

  if (!job_id) {
    return {
      status: "idle",
      result: undefined,
      modelProvider: undefined,
      progressInfo: undefined,
    };
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
      progressInfo: storedJob.progress_info(),
    };
  }

  try {
    // Poll for updates
    const updatedJob = await storedJob.poll();
    jobStore[job_id] = updatedJob;

    const result = updatedJob.result();
    const status = updatedJob.status();
    const modelProvider = updatedJob.provider();
    const progressInfo = updatedJob.progress_info();
    const loaderData = {
      status,
      result,
      modelProvider,
      progressInfo,
    };
    console.log(loaderData);

    return {
      status,
      result,
      modelProvider,
      progressInfo,
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
  let job;
  try {
    job = await launch_sft_job(validatedData);
  } catch (error) {
    const errors = {
      message:
        error instanceof Error
          ? error.message
          : "Unknown error occurred while launching fine-tuning job",
    };
    return data({ errors }, { status: 500 });
  }
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
  if (loaderData.status === "error") {
    return (
      <div className="text-red-500 text-sm">
        Error: {(loaderData as { error: string }).error}
      </div>
    );
  }
  const { status, result, modelProvider, progressInfo } = loaderData;
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
  const finalResult =
    result && modelProvider
      ? dump_model_config(get_fine_tuned_model_config(result, modelProvider))
      : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="p-4">
        <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">
          Supervised Fine-Tuning
        </h2>
        {status === "idle" && (
          <FineTuningForm
            config={config}
            submissionPhase={submissionPhase}
            setSubmissionPhase={setSubmissionPhase}
          />
        )}

        {finalResult && (
          <div className="p-4 bg-gray-100 rounded-lg mt-4">
            <div className="mb-2 font-medium">Configuration</div>
            <Textarea
              value={finalResult}
              className="w-full h-48 resize-none bg-transparent border-none focus:ring-0"
              readOnly
            />
          </div>
        )}

        {status !== "idle" && <ProgressIndicator progressInfo={progressInfo} />}
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
  const errorsOnSubmit = fetcher.data?.errors;
  if (errorsOnSubmit) {
    setSubmissionPhase("idle");
  }
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
      console.error("Submission error (likely a bug):", error);
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
              <p className="text-red-500 text-sm">{errors.function.message}</p>
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
              <p className="text-red-500 text-sm">{errors.metric.message}</p>
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
          {errorsOnSubmit && (
            <p className="text-red-500 text-sm">{errorsOnSubmit.message}</p>
          )}
        </form>
      </Form>
    </div>
  );
}
