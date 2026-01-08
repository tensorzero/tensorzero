import type { Route } from "./+types/route";
import { redirect, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
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
import { isInfraError } from "~/utils/tensorzero/errors";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");

  try {
    const [totalEvaluationRuns, evaluationRunsResponse] = await Promise.all([
      getTensorZeroClient().countEvaluationRuns(),
      getTensorZeroClient().listEvaluationRuns(limit, offset),
    ]);
    const evaluationRuns = evaluationRunsResponse.runs;

    return {
      totalEvaluationRuns,
      evaluationRuns,
      offset,
      limit,
    };
  } catch (error) {
    // Graceful degradation: return empty data on infra errors
    if (isInfraError(error)) {
      logger.warn("Infrastructure unavailable, showing degraded evaluations");
      return {
        totalEvaluationRuns: 0,
        evaluationRuns: [],
        offset,
        limit,
      };
    }
    throw error;
  }
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
