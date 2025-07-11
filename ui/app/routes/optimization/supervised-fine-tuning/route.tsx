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
  dump_provider_config,
  get_fine_tuned_provider_config,
} from "~/utils/config/models";
import type { Route } from "./+types/route";
import FineTuningStatus from "./FineTuningStatus";
import { SFTForm } from "./SFTForm";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { SFTFormValuesSchema } from "./types";
import { launch_sft_job } from "~/utils/supervised_fine_tuning/client";
import { Badge } from "~/components/ui/badge";
import { ModelBadge } from "~/components/model/ModelBadge";

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

  return redirect(
    `/optimization/supervised-fine-tuning/${validatedData.jobId}`,
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
      return () => {
        clearInterval(interval);
      };
    }
  }, [props, revalidator]);

  const finalResult =
    props.status === "completed"
      ? dump_provider_config(
          props.result,
          get_fine_tuned_provider_config(props.result, props.modelProvider),
        )
      : null;
  if (finalResult && submissionPhase !== "complete") {
    setSubmissionPhase("complete");
  }

  return props.status === "idle" ? (
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
      <PageHeader
        label="Supervised Fine-Tuning Job"
        heading={props.formData.jobId}
      >
        <div className="flex items-center gap-2">
          <Badge
            variant={
              props.status === "running"
                ? "default"
                : props.status === "completed"
                  ? "secondary"
                  : "destructive"
            }
          >
            {props.status}
          </Badge>
          <ModelBadge provider={props.modelProvider} />
        </div>
      </PageHeader>

      <FineTuningStatus status={props} result={finalResult} />
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
