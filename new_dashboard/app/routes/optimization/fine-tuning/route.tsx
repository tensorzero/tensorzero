import {
  useFetcher,
  type LoaderFunctionArgs,
  type MetaFunction,
} from "react-router";
import { useEffect, useState } from "react";
import { type SFTFormValues, SFTFormValuesSchema } from "./types";
import type { Route } from "./+types/route";
import { v7 as uuid } from "uuid";
import type { SFTJob } from "~/utils/fine_tuning/common";
import { useRevalidator } from "react-router";
import { redirect } from "react-router";
import { launch_sft_job } from "~/utils/fine_tuning/client";

export const meta: MetaFunction = () => {
  return [
    { title: "TensorZeroFine-Tuning Dashboard" },
    { name: "description", content: "Fine Tuning Optimization Dashboard" },
  ];
};

// Mutable store mapping job IDs to their info
export const jobStore: { [jobId: string]: SFTJob } = {};

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const job_id = url.searchParams.get("job_id");

  if (!job_id) {
    return { jobInfo: null, status: "idle" };
  }

  const storedJob = jobStore[job_id];
  if (!storedJob) {
    throw new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
    });
  }

  try {
    // Poll for updates
    console.log("polling for updates");
    const updatedJob = await storedJob.poll();
    console.log("updatedJob", updatedJob);
    jobStore[job_id] = updatedJob;

    const result = updatedJob.result();
    // TODO (Viraj, important!): fix the status here.
    const status = result ? "completed" : "running";

    return {
      jobInfo: updatedJob,
      status,
      result,
    };
  } catch (error) {
    return {
      jobInfo: storedJob,
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

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

  return redirect(`/optimization/fine-tuning?job_id=${validatedData.jobId}`);
}

export default function FineTuning({ loaderData }: Route.ComponentProps) {
  const { jobInfo, status, result, error } = loaderData;
  console.log("jobInfo in component", jobInfo);
  const revalidator = useRevalidator();
  let fetcher = useFetcher();

  useEffect(() => {
    if (status === "running") {
      const interval = setInterval(() => {
        revalidator.revalidate();
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [status, revalidator]);

  const testData: SFTFormValues = {
    function: "dashboard_fixture_extract_entities",
    metric: "dashboard_fixture_exact_match",
    model: {
      displayName: "llama-3.1-8b-instruct",
      name: "accounts/fireworks/models/llama-v3p1-8b-instruct",
      provider: "fireworks",
    },
    // model: {
    //   displayName: "gpt-4o-mini-2024-07-18",
    //   name: "gpt-4o-mini-2024-07-18",
    //   provider: "openai",
    // },
    variant: "baseline",
    validationSplitPercent: 20,
    maxSamples: 1000,
    threshold: 0.8,
    jobId: uuid(),
  };

  return (
    <div>
      <fetcher.Form method="POST">
        <input type="hidden" name="data" value={JSON.stringify(testData)} />
        <button type="submit">Submit Test Data</button>
      </fetcher.Form>

      {status === "running" && <div>Job in progress...</div>}
      {status === "completed" && <div>Job completed! Result: {result}</div>}
      {status === "error" && <div>Error: {error}</div>}

      {jobInfo && <pre>{JSON.stringify(jobInfo, null, 2)}</pre>}
    </div>
  );
}
