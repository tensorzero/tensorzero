import { NotFoundError, ErrorWithStatus } from "~/utils/error";

import { LoaderFunctionArgs } from "react-router";
import { BadRequestError } from "~/utils/error";
import { poll_sft_openai } from "~/utils/fine_tuning/openai";
import { OpenAISFTJob } from "~/utils/fine_tuning/openai.client";

export async function loader({ params }: LoaderFunctionArgs) {
  const { job_id } = params;
  if (!job_id) {
    const error = new BadRequestError("Job ID is required to poll OpenAI SFT");
    throw error;
  }

  try {
    const job_info = await poll_sft_openai(job_id);

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
