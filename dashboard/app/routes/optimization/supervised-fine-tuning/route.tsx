import { data, type MetaFunction } from "react-router";
import { useEffect, useState } from "react";
import { SFTFormValuesSchema } from "./types";
import type {
  SFTJob,
  SFTJobStatus,
} from "~/utils/supervised_fine_tuning/common";
import { useRevalidator } from "react-router";
import { redirect } from "react-router";
import { launch_sft_job } from "~/utils/supervised_fine_tuning/client";
import { useConfig } from "~/context/config";
import {
  dump_model_config,
  get_fine_tuned_model_config,
  type ProviderType,
} from "~/utils/config/models";
import type { Route } from "./+types/route";
import FineTuningStatus from "./FineTuningStatus";
import { SFTResult } from "./SFTResult";
import { SFTForm } from "./SFTForm";

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
function get_progress_fixture(provider: ProviderType): SFTJobStatus {
  // 25% chance of returning an error
  if (Math.random() < 0.8) {
    return {
      status: "running",
      modelProvider: provider,
      formData: {
        function: "dashboard_fixture_write_haiku",
        metric: "dashboard_fixture_haiku_rating",
        jobId: "01943d5f-d649-7e0c-95b7-04b7944128ea",
        model: {
          displayName: "gpt-4o-mini-2024-07-18",
          name: "gpt-4o-mini-2024-07-18",
          provider: "openai",
        },
        variant: "baseline",
        validationSplitPercent: 20,
        maxSamples: 94,
        threshold: 0.5,
      },
      jobUrl:
        provider === "openai"
          ? "https://platform.openai.com/finetune/ftjob-abc123"
          : "https://fireworks.ai/dashboard/fine-tuning/ftjob-abc123",
      rawData: {
        status: "error",
        message: "Simulated error occurred during fine-tuning",
      },
    };
  }

  switch (provider) {
    case "openai":
      return {
        status: "running",
        modelProvider: "openai",
        formData: {
          function: "dashboard_fixture_write_haiku",
          metric: "dashboard_fixture_haiku_rating",
          model: {
            displayName: "gpt-4o-mini-2024-07-18",
            name: "gpt-4o-mini-2024-07-18",
            provider: "openai",
          },
          variant: "baseline",
          validationSplitPercent: 20,
          maxSamples: 94,
          threshold: 0.5,
          jobId: "01944200-d290-706d-b2f2-ca958d7ced80",
        },
        rawData: {
          status: "ok",
          info: {
            object: "fine_tuning.job",
            id: "ftjob-eC8vFeECwiVKNrjDxNrBkIvH",
            model: "gpt-4o-mini-2024-07-18",
            created_at: 1736274146,
            finished_at: null,
            fine_tuned_model: null,
            organization_id: "org-fewHWgmYjDeYGco5co60C7fh",
            result_files: [],
            status: "validating_files",
            validation_file: "file-EsNbLPMX57KNArvLqyMuph",
            training_file: "file-NoXMdRuTTQq5X7vRB95TJ8",
            hyperparameters: [Object],
            trained_tokens: null,
            error: {},
            user_provided_suffix: null,
            seed: 1683600021,
            estimated_finish: null,
            integrations: [],
            method: [Object],
          },
        },
        estimatedCompletionTime: new Date(Date.now() + 15 * 60 * 1000),
        jobUrl:
          "https://platform.openai.com/finetune/ftjob-eC8vFeECwiVKNrjDxNrBkIvH",
      };
    case "fireworks":
      return {
        status: "running",
        modelProvider: "fireworks",
        formData: {
          function: "dashboard_fixture_extract_entities",
          metric: "dashboard_fixture_exact_match",
          model: {
            displayName: "llama-3.1-8b-instruct",
            name: "accounts/fireworks/models/llama-v3p1-8b-instruct",
            provider: "fireworks",
          },
          variant: "baseline",
          validationSplitPercent: 20,
          maxSamples: 41,
          threshold: 0.5,
          jobId: "019441ec-cecb-7489-b59e-a2aa2ee942c1",
        },
        jobUrl:
          "https://fireworks.ai/dashboard/fine-tuning/c3b0372cb74e4155ba2811ca3c41e0bb",
        rawData: {
          status: "ok",
          info: {
            state: "RUNNING",
            modelId: "",
            baseModel: "accounts/fireworks/models/llama-v3p1-8b-instruct",
            batchSize: 16,
            createTime: "2025-01-07T18:00:35.637282Z",
            createdBy: "viraj@tensorzero.com",
            dataset:
              "accounts/viraj-ebfe5a/datasets/019441ec-ed15-73bc-8cf7-a1ef8533dcd8",
            evaluationSplit: 0.2,
            evaluation: false,
            evaluationDataset: "",
            learningRate: 0.0001,
            loraRank: 8,
            loraTargetModules: [],
            maskToken: "",
            microBatchSize: 0,
            name: "accounts/viraj-ebfe5a/fineTuningJobs/c3b0372cb74e4155ba2811ca3c41e0bb",
            padToken: "",
            status: [Object],
          },
        },
      };
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}

// If there is a job_id in the URL, grab it from the job store and pull it.
export async function loader({
  params,
}: Route.LoaderArgs): Promise<SFTJobStatus> {
  // for debugging ProgressIndicator without starting a real job
  // return get_progress_fixture("openai");
  const job_id = params.job_id;

  if (!job_id) {
    return {
      status: "idle",
    };
  }

  const storedJob = jobStore[job_id];
  if (!storedJob) {
    throw new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
    });
  }

  // Poll for updates
  const updatedJob = await storedJob.poll();
  jobStore[job_id] = updatedJob;
  const status = updatedJob.status();
  console.log("status", status);
  return status;
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
      <div className="text-sm text-red-500">Error: {loaderData.error}</div>
    );
  }
  const status = loaderData;
  const revalidator = useRevalidator();

  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");

  // If running, periodically poll for updates on the job
  useEffect(() => {
    if (status.status === "running") {
      setSubmissionPhase("pending");
      const interval = setInterval(() => {
        revalidator.revalidate();
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [status, revalidator]);

  const finalResult =
    status.status === "completed"
      ? dump_model_config(
          get_fine_tuned_model_config(status.result, status.modelProvider),
        )
      : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <main>
        <h2 className="mb-4 text-2xl font-semibold">Supervised Fine-Tuning</h2>
        <div className="mb-6 h-px w-full bg-gray-200"></div>
        {status.status === "idle" && (
          <SFTForm
            config={config}
            submissionPhase={submissionPhase}
            setSubmissionPhase={setSubmissionPhase}
          />
        )}

        {<FineTuningStatus status={status} />}
        <SFTResult finalResult={finalResult} />
      </main>
    </div>
  );
}
