import { poll_fine_tuning_job } from "~/utils/fine_tuning/openai.server";
import { json, type LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ params }: LoaderFunctionArgs) {
  const { job_id } = params;
  console.log("job_id", job_id);
  if (!job_id) {
    return json({ error: "Job ID is required" }, { status: 400 });
  }

  try {
    const job = await poll_fine_tuning_job(job_id);
    console.log("Polled fine-tuning job:", job?.status);

    if (job) {
      console.log(job);
      console.log(job.fine_tuned_model);
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
