import { poll_fine_tuning_job } from "~/utils/fine_tuning/openai.server";
import { type LoaderFunctionArgs } from "react-router";

export async function loader({ params }: LoaderFunctionArgs) {
  const { job_id } = params;
  if (!job_id) {
    return Response.json({ error: "Job ID is required" }, { status: 400 });
  }

  try {
    const job = await poll_fine_tuning_job(job_id);

    if (job) {
      return Response.json({
        status: job.status,
        fine_tuned_model: job.fine_tuned_model,
        job: job,
      });
    } else {
      return Response.json({ error: "Job not found" }, { status: 404 });
    }
  } catch (error) {
    return Response.json({ error: (error as Error).message }, { status: 500 });
  }
}
