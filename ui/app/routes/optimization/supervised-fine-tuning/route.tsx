import { data, type RouteHandle } from "react-router";
import { useEffect, useState } from "react";
import type {
  SFTJob,
  SFTJobStatus,
} from "~/utils/supervised_fine_tuning/common";
import { useRevalidator } from "react-router";
import { redirect } from "react-router";
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
import { SFTFormValuesSchema } from "./types";
import { launch_sft_job } from "~/utils/supervised_fine_tuning/client";

export const handle: RouteHandle = {
  crumb: () => ["Supervised Fine-Tuning"],
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

  // The query parameter is currently just used by e2e tests to check that we're
  // using the expected backend
  return redirect(
    `/optimization/supervised-fine-tuning/${validatedData.jobId}?backend=nodejs`,
  );
}

type LoaderData = Route.ComponentProps["loaderData"];

function SupervisedFineTuningImpl(
  props: LoaderData & {
    status: "running" | "completed" | "idle";
  },
) {
  const config = useConfig();
  const revalidator = useRevalidator();

  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");

  // If running, periodically poll for updates on the job
  useEffect(() => {
    if (props.status === "running") {
      setSubmissionPhase("pending");
      const interval = setInterval(
        () => {
          revalidator.revalidate();
        },
        navigator.userAgent === "TensorZeroE2E" ? 500 : 10000,
      );
      return () => clearInterval(interval);
    }
  }, [props, revalidator]);

  const finalResult =
    props.status === "completed"
      ? dump_model_config(
          get_fine_tuned_model_config(props.result, props.modelProvider),
        )
      : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return (
    <PageLayout>
      <PageHeader heading="Supervised Fine-Tuning" />
      <SectionLayout>
        {props.status === "idle" && (
          <SFTForm
            config={config}
            submissionPhase={submissionPhase}
            setSubmissionPhase={setSubmissionPhase}
          />
        )}
        <FineTuningStatus status={props} />
        <SFTResult finalResult={finalResult} />
      </SectionLayout>
    </PageLayout>
  );
}

// Renders the fine-tuning form and status info.
export default function SupervisedFineTuning(props: Route.ComponentProps) {
  const { loaderData } = props;
  if (loaderData.status === "error") {
    return (
      <PageLayout>
        <PageHeader heading="Supervised Fine-Tuning" />
        <SectionLayout>
          <div className="text-sm text-red-500">Error: {loaderData.error}</div>
        </SectionLayout>
      </PageLayout>
    );
  }
  return <SupervisedFineTuningImpl {...loaderData} />;
}
