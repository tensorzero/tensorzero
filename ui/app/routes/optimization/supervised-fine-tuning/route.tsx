import { data, type RouteHandle } from "react-router";
import { useEffect, useState } from "react";
import { useRevalidator } from "react-router";
import { redirect } from "react-router";
import { useConfig } from "~/context/config";
import { dump_optimizer_output } from "~/utils/config/models";
import type { Route } from "./+types/route";
import FineTuningStatus from "./FineTuningStatus";
import { SFTForm } from "./SFTForm";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { SFTFormValuesSchema, type SFTFormValues } from "./types";
import {
  launch_sft_job,
  poll_sft_job,
} from "~/utils/supervised_fine_tuning/client";
import { Badge } from "~/components/ui/badge";
import { ModelBadge } from "~/components/model/ModelBadge";
import type {
  OptimizationJobHandle,
  OptimizationJobInfo,
} from "tensorzero-node";
import { toSupervisedFineTuningJobUrl } from "~/utils/urls";
import { protectAction } from "~/utils/read-only.server";

export const handle: RouteHandle = {
  crumb: () => ["Supervised Fine-Tuning"],
};

export interface JobInfo {
  formData: SFTFormValues;
  handle: OptimizationJobHandle;
}

// Mutable store mapping job IDs to their info
export const jobStore: { [jobId: string]: JobInfo } = {};

// If there is a job_id in the URL, grab it from the job store and pull it.
export async function loader({ params }: Route.LoaderArgs): Promise<{
  jobInfo: { status: "idle" } | OptimizationJobInfo;
  formData: SFTFormValues | null;
  jobHandle: OptimizationJobHandle | null;
}> {
  // for debugging ProgressIndicator without starting a real job
  const job_id = params.job_id;

  if (!job_id) {
    return {
      jobInfo: { status: "idle" },
      formData: null,
      jobHandle: null,
    };
  }

  const storedJob = jobStore[job_id];
  if (!storedJob) {
    throw new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
    });
  }

  // Poll for updates
  const status = await poll_sft_job(storedJob.handle);
  return {
    jobInfo: status,
    formData: storedJob.formData,
    jobHandle: storedJob.handle,
  };
}

// The action actually launches the fine-tuning job.
export async function action({ request }: Route.ActionArgs) {
  protectAction();
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
  jobStore[validatedData.jobId] = {
    formData: validatedData,
    handle: job,
  };

  return redirect(toSupervisedFineTuningJobUrl(validatedData.jobId));
}

type LoaderData = Route.ComponentProps["loaderData"];

function SupervisedFineTuningImpl(props: LoaderData) {
  const { jobInfo, formData, jobHandle } = props;
  const config = useConfig();
  const revalidator = useRevalidator();

  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");

  // If running, periodically poll for updates on the job
  useEffect(() => {
    if (jobInfo.status === "pending") {
      setSubmissionPhase("pending");
      const interval = setInterval(
        () => {
          revalidator.revalidate();
        },
        navigator.userAgent === "TensorZeroE2E" ? 500 : 10000,
      );
      return () => {
        clearInterval(interval);
      };
    }
    return undefined;
  }, [jobInfo, revalidator]);

  const finalResult =
    jobInfo.status === "completed"
      ? // TODO (Viraj, now): fix up the optimizer output to match E2E test
        dump_optimizer_output(jobInfo.output)
      : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return jobInfo.status === "idle" || !formData || !jobHandle ? (
    <PageLayout>
      <PageHeader heading="Supervised Fine-Tuning" />
      <SectionLayout>
        <SFTForm
          config={config}
          submissionPhase={submissionPhase}
          setSubmissionPhase={setSubmissionPhase}
        />
      </SectionLayout>
    </PageLayout>
  ) : (
    <PageLayout>
      <PageHeader label="Supervised Fine-Tuning Job" heading={formData?.jobId}>
        <div className="flex items-center gap-2">
          <Badge
            variant={
              jobInfo.status === "pending"
                ? "default"
                : jobInfo.status === "completed"
                  ? "secondary"
                  : "destructive"
            }
          >
            {jobInfo.status === "pending" ? "running" : jobInfo.status}
          </Badge>
          {formData?.model.provider && (
            <ModelBadge provider={formData.model.provider} />
          )}
        </div>
      </PageHeader>

      <FineTuningStatus
        status={jobInfo}
        formData={formData ?? {}}
        result={finalResult}
        jobHandle={jobHandle}
      />
    </PageLayout>
  );
}

// Renders the fine-tuning form and status info.
export default function SupervisedFineTuning(props: Route.ComponentProps) {
  const { loaderData } = props;
  if (loaderData.jobInfo.status === "failed") {
    return (
      <PageLayout>
        <PageHeader heading="Supervised Fine-Tuning" />
        <SectionLayout>
          <div className="text-sm text-red-500">
            {JSON.stringify(loaderData.jobInfo.error, null, 2)}
          </div>
        </SectionLayout>
      </PageLayout>
    );
  }
  return <SupervisedFineTuningImpl {...loaderData} />;
}
