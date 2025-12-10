import type { Route } from "./+types/route";
import { isRouteErrorResponse, redirect, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { getEvaluationRunInfo } from "~/utils/clickhouse/evaluations.server";
import EvaluationRunsTable from "./EvaluationRunsTable";
import { useState } from "react";
import { EvaluationsActions } from "./EvaluationsActions";
import LaunchEvaluationModal from "./LaunchEvaluationModal";
import {
  parseEvaluationFormData,
  runEvaluation,
} from "~/utils/evaluations.server";
import { logger } from "~/utils/logger";
import { toEvaluationUrl } from "~/utils/urls";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const totalEvaluationRuns = await getTensorZeroClient().countEvaluationRuns();
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");
  const evaluationRuns = await getEvaluationRunInfo(limit, offset);

  return {
    totalEvaluationRuns,
    evaluationRuns,
    offset,
    limit,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const evaluationFormData = parseEvaluationFormData({
    evaluation_name: formData.get("evaluation_name"),
    dataset_name: formData.get("dataset_name"),
    variant_name: formData.get("variant_name"),
    concurrency_limit: formData.get("concurrency_limit"),
    inference_cache: formData.get("inference_cache"),
    max_datapoints: formData.get("max_datapoints"),
    precision_targets: formData.get("precision_targets"),
  });

  if (!evaluationFormData) {
    throw new Response("Invalid form data", { status: 400 });
  }

  const {
    evaluation_name,
    dataset_name,
    variant_name,
    concurrency_limit,
    inference_cache,
    max_datapoints,
    precision_targets,
  } = evaluationFormData;

  let evaluation_start_info;
  try {
    evaluation_start_info = await runEvaluation(
      evaluation_name,
      dataset_name,
      variant_name,
      concurrency_limit,
      inference_cache,
      max_datapoints,
      precision_targets,
    );
  } catch (error) {
    logger.error("Error starting evaluation:", error);
    throw new Response(`Failed to start evaluation: ${error}`, {
      status: 500,
    });
  }
  return redirect(
    toEvaluationUrl(evaluation_name, {
      evaluation_run_ids: evaluation_start_info.evaluation_run_id,
    }),
  );
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const { totalEvaluationRuns, evaluationRuns, offset, limit } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - limit));
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
          disableNext={offset + limit >= totalEvaluationRuns}
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
