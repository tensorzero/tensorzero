import { poll_fine_tuning_job } from "~/utils/fine_tuning/openai.server";
import { json, type LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ params }: LoaderFunctionArgs) {
  const { job_id } = params;
  if (!job_id) {
    return json({ error: "Job ID is required" }, { status: 400 });
  }

  try {
    const job = await poll_fine_tuning_job(job_id);

    if (job) {
      return json({
        status: job.status,
        fine_tuned_model: job.fine_tuned_model,
        job: job,
      });
    } else {
      return json({ error: "Job not found" }, { status: 404 });
    }
  } catch (error) {
    return json({ error: (error as Error).message }, { status: 500 });
  }
}
