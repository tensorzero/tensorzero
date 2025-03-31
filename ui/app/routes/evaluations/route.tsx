import type { Route } from "./+types/route";
import { isRouteErrorResponse, redirect, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import {
  countTotalEvaluationRuns,
  getEvaluationRunInfo,
} from "~/utils/clickhouse/evaluations.server";
import { getConfig } from "~/utils/config/index.server";
import EvalRunsTable from "./EvalRunsTable";
import { useState } from "react";
import { EvaluationsActions } from "./EvaluationsActions";
import LaunchEvalModal from "./LaunchEvalModal";
import { runEvaluation } from "~/utils/evaluations.server";

export async function loader({ request }: Route.LoaderArgs) {
  const totalEvalRuns = await countTotalEvaluationRuns();
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "15");
  const evalRuns = await getEvaluationRunInfo(pageSize, offset);
  const config = await getConfig();
  const evalRunsWithDataset = evalRuns.map((runInfo) => {
    const dataset = config.evaluations[runInfo.eval_name].dataset_name;
    return {
      ...runInfo,
      dataset,
    };
  });

  return {
    totalEvalRuns,
    evalRunsWithDataset,
    offset,
    pageSize,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const eval_name = formData.get("eval_name");
  const variant_name = formData.get("variant_name");
  const concurrency_limit = formData.get("concurrency_limit");
  let eval_start_info;
  try {
    eval_start_info = await runEvaluation(
      eval_name as string,
      variant_name as string,
      parseInt(concurrency_limit as string),
    );
  } catch (error) {
    console.error("Error starting evaluation:", error);
    throw new Response("Failed to start eval", { status: 500 });
  }
  return redirect(
    `/evaluations/${eval_name}?eval_run_ids=${eval_start_info.eval_run_id}`,
  );
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const { totalEvalRuns, evalRunsWithDataset, offset, pageSize } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const [launchEvalModalIsOpen, setLaunchEvalModalIsOpen] = useState(false);

  return (
    <PageLayout>
      <PageHeader heading="Evaluation Runs" count={totalEvalRuns} />
      <SectionLayout>
        <EvaluationsActions onNewRun={() => setLaunchEvalModalIsOpen(true)} />
        <EvalRunsTable evalRuns={evalRunsWithDataset} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + pageSize >= totalEvalRuns}
        />
      </SectionLayout>
      <LaunchEvalModal
        isOpen={launchEvalModalIsOpen}
        onClose={() => setLaunchEvalModalIsOpen(false)}
      />
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
