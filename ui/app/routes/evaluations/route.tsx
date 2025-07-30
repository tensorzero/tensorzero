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
import EvaluationRunsTable from "./EvaluationRunsTable";
import { useState } from "react";
import { EvaluationsActions } from "./EvaluationsActions";
import LaunchEvaluationModal from "./LaunchEvaluationModal";
import {
  runEvaluation,
  type InferenceCacheSetting,
} from "~/utils/evaluations.server";
import { logger } from "~/utils/logger";

export async function loader({ request }: Route.LoaderArgs) {
  const totalEvaluationRuns = await countTotalEvaluationRuns();
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "15");
  const evaluationRuns = await getEvaluationRunInfo(pageSize, offset);

  return {
    totalEvaluationRuns,
    evaluationRuns,
    offset,
    pageSize,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  console.log(`Launching evaluation with: ${JSON.stringify(formData)}`);
  const evaluation_name = formData.get("evaluation_name");
  const dataset_name = formData.get("dataset_name");
  const variant_name = formData.get("variant_name");
  const concurrency_limit = formData.get("concurrency_limit");
  const inference_cache = formData.get("inference_cache");
  let evaluation_start_info;
  try {
    evaluation_start_info = await runEvaluation(
      evaluation_name as string,
      dataset_name as string,
      variant_name as string,
      parseInt(concurrency_limit as string),
      inference_cache as InferenceCacheSetting,
    );
  } catch (error) {
    logger.error("Error starting evaluation:", error);
    throw new Response(`Failed to start evaluation: ${error}`, {
      status: 500,
    });
  }
  return redirect(
    `/evaluations/${evaluation_name}?evaluation_run_ids=${evaluation_start_info.evaluation_run_id}`,
  );
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const { totalEvaluationRuns, evaluationRuns, offset, pageSize } = loaderData;

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
  const [launchEvaluationModalIsOpen, setLaunchEvaluationModalIsOpen] =
    useState(false);

  return (
    <PageLayout>
      <PageHeader heading="Evaluation Runs" count={totalEvaluationRuns} />
      <SectionLayout>
        <EvaluationsActions
          onNewRun={() => setLaunchEvaluationModalIsOpen(true)}
        />
        <EvaluationRunsTable evaluationRuns={evaluationRuns} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + pageSize >= totalEvaluationRuns}
        />
      </SectionLayout>
      <LaunchEvaluationModal
        isOpen={launchEvaluationModalIsOpen}
        onClose={() => setLaunchEvaluationModalIsOpen(false)}
      />
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

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
