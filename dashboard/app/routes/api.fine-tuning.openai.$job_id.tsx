import { NotFoundError, ErrorWithStatus } from "~/utils/error";

import { LoaderFunctionArgs } from "react-router";
import { BadRequestError } from "~/utils/error";
import { client } from "~/utils/fine_tuning/openai";
import { OpenAISFTJob } from "~/utils/fine_tuning/client";

export async function loader({ params }: LoaderFunctionArgs) {
  const { job_id } = params;
  if (!job_id) {
    const error = new BadRequestError("Job ID is required to poll OpenAI SFT");
    throw error;
  }

  console.log("job_id", job_id);
  try {
    const job_info = await poll_sft_openai(job_id);
    console.log("job_info", job_info);

    if (job_info) {
      return Response.json(
        new OpenAISFTJob(
          job_info.id,
          job_info.status,
          job_info.fine_tuned_model ?? undefined,
        ),
      );
    } else {
      throw new NotFoundError("Job not found");
    }
  } catch (error) {
    return Response.json(
      { error: (error as Error).message },
      { status: error instanceof ErrorWithStatus ? error.status : 500 },
    );
  }
}

async function poll_sft_openai(job_id: string) {
  if (!job_id) {
    const error = new BadRequestError("Job ID is required to poll OpenAI SFT");
    throw error;
  }

  const job = await client.fineTuning.jobs.retrieve(job_id);
  return job;
}
