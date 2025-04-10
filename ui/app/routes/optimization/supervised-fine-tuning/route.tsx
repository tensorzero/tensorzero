import { data, type MetaFunction } from "react-router";
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
import { SFTFormValuesSchema, type SFTFormValues } from "./types";
import { launch_sft_job } from "~/utils/supervised_fine_tuning/client";
import { FF_ENABLE_PYTHON } from "./featureflag.server";

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

  if (FF_ENABLE_PYTHON) {
    return await loadPythonFineTuneJob(job_id);
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

async function loadPythonFineTuneJob(job_id: string) {
  const res = await fetch(`http://localhost:7000/optimizations/poll/${job_id}`);
  if (!res.ok) {
    if (res.status === 404) {
      throw new Response(JSON.stringify({ error: await res.text() }), {
        status: 404,
      });
    } else {
      throw new Response(JSON.stringify({ error: await res.text() }), {
        status: 500,
      });
    }
  }
  return await res.json();
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

  if (FF_ENABLE_PYTHON) {
    return await startPythonFineTune(jsonData, validatedData);
  }
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

async function startPythonFineTune(
  parsedFormData: object,
  validatedData: SFTFormValues,
) {
  try {
    const res = await fetch("http://localhost:7000/optimizations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: {
          kind: "sft",
          ...parsedFormData,
        },
      }),
    });
    const resText = await res.text();
    if (!res.ok) {
      return data(
        { message: `Error ${res.status} from fine-tuning server: ${resText}` },
        { status: 500 },
      );
    }
  } catch (error) {
    const errors = {
      message:
        error instanceof Error
          ? error.message
          : "Unknown error occurred while launching fine-tuning job",
    };
    return data({ errors }, { status: 500 });
  }

  return redirect(
    `/optimization/supervised-fine-tuning/${validatedData.jobId}?backend=python`,
  );
}

// Renders the fine-tuning form and status info.
export default function SupervisedFineTuning({
  loaderData,
}: Route.ComponentProps) {
  const config = useConfig();
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
        <FineTuningStatus status={status} />
        <SFTResult finalResult={finalResult} />
      </SectionLayout>
    </PageLayout>
  );
}
