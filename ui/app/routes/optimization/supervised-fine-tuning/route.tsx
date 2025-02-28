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
} from "~/utils/config/models";
import type { Route } from "./+types/route";
import FineTuningStatus from "./FineTuningStatus";
import { SFTResult } from "./SFTResult";
import { SFTForm } from "./SFTForm";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";

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
export async function loader({
  params,
}: Route.LoaderArgs): Promise<SFTJobStatus> {
  // for debugging ProgressIndicator without starting a real job
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
      <div className="container mx-auto px-4 pb-8">
        <PageLayout>
          <PageHeader heading="Supervised Fine-Tuning" />
          <SectionLayout>
            <div className="text-sm text-red-500">
              Error: {loaderData.error}
            </div>
          </SectionLayout>
        </PageLayout>
      </div>
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
    <div className="container mx-auto px-4 pb-8">
      <PageLayout>
        <PageHeader heading="Supervised Fine-Tuning" />
        <SectionLayout>
          {status.status === "idle" && (
            <SFTForm
              config={config}
              submissionPhase={submissionPhase}
              setSubmissionPhase={setSubmissionPhase}
            />
          )}

          {<FineTuningStatus status={status} />}
          <SFTResult finalResult={finalResult} />
        </SectionLayout>
      </PageLayout>
    </div>
  );
}
