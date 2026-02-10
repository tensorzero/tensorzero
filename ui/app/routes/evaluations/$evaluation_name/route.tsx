import { Suspense, useCallback, useEffect, useState } from "react";
import { AlertCircle, Loader2, StopCircle } from "lucide-react";
import { Button } from "~/components/ui/button";
import type { Route } from "./+types/route";
import type { EvaluationResultRow } from "~/types/tensorzero";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import {
  getEvaluationResults,
  pollForEvaluationResults,
} from "~/utils/clickhouse/evaluations.server";
import { getEvaluatorMetricName } from "~/utils/clickhouse/evaluations";
import { EvaluationTable, type SelectedRowData } from "./EvaluationTable";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import {
  Await,
  data,
  isRouteErrorResponse,
  redirect,
  useAsyncError,
  useLocation,
  useNavigate,
  useFetcher,
} from "react-router";
import AutoRefreshIndicator, { useAutoRefresh } from "./AutoRefreshIndicator";
import BasicInfo from "./EvaluationBasicInfo";
import { useConfig } from "~/context/config";
import { getRunningEvaluation } from "~/utils/evaluations.server";
import {
  EvaluationErrorInfo,
  type EvaluationErrorDisplayInfo,
} from "./EvaluationErrorInfo";
import {
  addEvaluationHumanFeedback,
  getTensorZeroClient,
} from "~/utils/tensorzero.server";
import { useToast } from "~/hooks/use-toast";
import { logger } from "~/utils/logger";
import { ActionBar } from "~/components/layout/ActionBar";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { DatasetSelect } from "~/components/dataset/DatasetSelect";
import { handleBulkAddToDataset } from "./bulkAddToDataset.server";
import { useBulkAddToDatasetToast } from "./useBulkAddToDatasetToast";
import { useReadOnly } from "~/context/read-only";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import type { ShouldRevalidateFunctionArgs } from "react-router";

// Prevent fetcher submissions (e.g. kill evaluation) from triggering
// a full loader revalidation. The auto-refresh interval handles updates.
export function shouldRevalidate({
  formAction,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  if (formAction?.includes("/kill")) {
    return false;
  }
  return defaultShouldRevalidate;
}

type EvaluationData = {
  selected_evaluation_run_infos: Awaited<
    ReturnType<ReturnType<typeof getTensorZeroClient>["getEvaluationRunInfos"]>
  >["run_infos"];
  evaluation_results: EvaluationResultRow[];
  evaluation_statistics: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getEvaluationStatistics"]
    >
  >["statistics"];
  total_datapoints: number;
};

async function fetchEvaluationData(
  evaluation_name: string,
  function_name: string,
  function_type: "chat" | "json",
  selected_evaluation_run_ids_array: string[],
  metric_names: string[],
  limit: number,
  offset: number,
  newFeedbackId: string | null,
): Promise<EvaluationData> {
  if (selected_evaluation_run_ids_array.length === 0) {
    return {
      selected_evaluation_run_infos: [],
      evaluation_results: [],
      evaluation_statistics: [],
      total_datapoints: 0,
    };
  }

  const tensorZeroClient = getTensorZeroClient();

  const evaluationRunInfosPromise = tensorZeroClient
    .getEvaluationRunInfos(selected_evaluation_run_ids_array, function_name)
    .then((response) => response.run_infos);

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the evaluation results as it is eventually consistent.
  // In this case, we poll for the evaluation results until the feedback is found.
  const resultsPromise: Promise<EvaluationResultRow[]> = newFeedbackId
    ? pollForEvaluationResults(
        evaluation_name,
        selected_evaluation_run_ids_array,
        newFeedbackId,
        limit,
        offset,
      )
    : getEvaluationResults(
        evaluation_name,
        selected_evaluation_run_ids_array,
        limit,
        offset,
      );

  const statisticsPromise = tensorZeroClient
    .getEvaluationStatistics(
      function_name,
      function_type,
      metric_names,
      selected_evaluation_run_ids_array,
    )
    .then((response) => response.statistics);

  const totalDatapointsPromise = tensorZeroClient.countDatapointsForEvaluation(
    function_name,
    selected_evaluation_run_ids_array,
  );

  const [
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    total_datapoints,
  ] = await Promise.all([
    evaluationRunInfosPromise,
    resultsPromise,
    statisticsPromise,
    totalDatapointsPromise,
  ]);

  return {
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    total_datapoints,
  };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const evaluationConfig = config.evaluations[params.evaluation_name];
  if (!evaluationConfig) {
    throw data(
      `Evaluation config not found for evaluation ${params.evaluation_name}`,
      { status: 404 },
    );
  }
  const function_name = evaluationConfig.function_name;
  const functionConfig = await getFunctionConfig(function_name, config);
  const function_type = functionConfig?.type;
  if (!function_type) {
    throw data(`Function config not found for function ${function_name}`, {
      status: 404,
    });
  }

  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const newFeedbackId = searchParams.get("newFeedbackId");
  const newJudgeDemonstrationId = searchParams.get("newJudgeDemonstrationId");

  const selected_evaluation_run_ids = searchParams.get("evaluation_run_ids");
  const selected_evaluation_run_ids_array = selected_evaluation_run_ids
    ? selected_evaluation_run_ids.split(",")
    : [];

  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");

  const evaluator_names = Object.keys(evaluationConfig.evaluators);
  const metric_names = evaluator_names.map((evaluatorName) =>
    getEvaluatorMetricName(params.evaluation_name, evaluatorName),
  );

  const running_evaluation_run_ids = selected_evaluation_run_ids_array.filter(
    (evaluationRunId) => {
      const runningEvaluation = getRunningEvaluation(evaluationRunId);
      if (!runningEvaluation) {
        return false;
      }
      if (runningEvaluation.completed) {
        // If the evaluation completed more than 5 seconds ago, consider it done
        if (runningEvaluation.completed.getTime() + 5000 < Date.now()) {
          return false;
        }
      }
      return true;
    },
  );

  const errors: Record<string, EvaluationErrorDisplayInfo> =
    selected_evaluation_run_ids_array.reduce(
      (acc, evaluationRunId) => {
        const evaluationRunInfo = getRunningEvaluation(evaluationRunId);
        if (evaluationRunInfo?.errors) {
          acc[evaluationRunId] = {
            variantName: evaluationRunInfo.variantName,
            errors: evaluationRunInfo.errors,
          };
        } else {
          acc[evaluationRunId] = {
            variantName: evaluationRunId,
            errors: [],
          };
        }
        return acc;
      },
      {} as Record<string, EvaluationErrorDisplayInfo>,
    );

  return {
    evaluation_name: params.evaluation_name,
    evaluationData: fetchEvaluationData(
      params.evaluation_name,
      function_name,
      function_type,
      selected_evaluation_run_ids_array,
      metric_names,
      limit,
      offset,
      newFeedbackId,
    ),
    has_selected_runs: selected_evaluation_run_ids_array.length > 0,
    offset,
    limit,
    evaluator_names,
    running_evaluation_run_ids,
    errors,
    newFeedbackId,
    newJudgeDemonstrationId,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const _action = formData.get("_action");
  switch (_action) {
    case "addFeedback": {
      const response = await addEvaluationHumanFeedback(formData);
      const url = new URL(request.url);
      url.searchParams.delete("beforeFeedback");
      url.searchParams.delete("afterFeedback");
      url.searchParams.set(
        "newFeedbackId",
        response.feedbackResponse.feedback_id,
      );
      if (response.judgeDemonstrationResponse) {
        url.searchParams.set(
          "newJudgeDemonstrationId",
          response.judgeDemonstrationResponse.feedback_id,
        );
      } else {
        logger.warn("No judge demonstration response");
      }
      return redirect(url.toString());
    }
    case "addMultipleToDataset": {
      const dataset = formData.get("dataset");
      const selectedItemsJson = formData.get("selectedItems");
      const evaluation_name = formData.get("evaluation_name");

      if (!dataset || !selectedItemsJson || !evaluation_name) {
        return data(
          { error: "Missing required fields", success: false },
          { status: 400 },
        );
      }

      try {
        const selectedItems = JSON.parse(selectedItemsJson.toString());
        return await handleBulkAddToDataset(
          dataset.toString(),
          selectedItems,
          evaluation_name.toString(),
        );
      } catch (error) {
        logger.error("Error processing bulk add to dataset:", error);
        return data(
          { error: "Failed to process request", success: false },
          { status: 500 },
        );
      }
    }
    case null:
      logger.error("No action provided");
      return data({ error: "No action provided" }, { status: 400 });
    default:
      logger.error(`Unknown action: ${_action}`);
      return data({ error: `Unknown action: ${_action}` }, { status: 400 });
  }
}

function ResultsSkeleton() {
  return (
    <>
      <div className="flex items-center">
        <SectionHeader heading="Results" />
      </div>
      <Skeleton className="h-64 w-full" />
      <Skeleton className="mt-4 h-10 w-48" />
    </>
  );
}

function ResultsError() {
  const error = useAsyncError();
  let message = "Failed to load evaluation results";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading evaluation results"
      description={message}
    />
  );
}

function ResultsContent({
  evaluation_name,
  data,
  evaluator_names,
  any_evaluation_is_running,
  has_selected_runs,
  offset,
  limit,
  selectedRows,
  setSelectedRows,
  onKill,
  isKilling,
}: {
  evaluation_name: string;
  data: EvaluationData;
  evaluator_names: string[];
  any_evaluation_is_running: boolean;
  has_selected_runs: boolean;
  offset: number;
  limit: number;
  selectedRows: Map<string, SelectedRowData>;
  setSelectedRows: React.Dispatch<
    React.SetStateAction<Map<string, SelectedRowData>>
  >;
  onKill: () => void;
  isKilling: boolean;
}) {
  const navigate = useNavigate();
  const {
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    total_datapoints,
  } = data;

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

  return (
    <>
      <div className="flex items-center">
        <SectionHeader heading="Results" />
        <div
          className="ml-4 flex items-center gap-4"
          data-testid="auto-refresh-wrapper"
          data-running={any_evaluation_is_running}
        >
          <AutoRefreshIndicator isActive={any_evaluation_is_running} />
          {any_evaluation_is_running && (
            <Button
              variant="outline"
              size="sm"
              onClick={onKill}
              disabled={isKilling}
              slotLeft={
                isKilling ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <StopCircle className="h-3.5 w-3.5" />
                )
              }
            >
              Stop
            </Button>
          )}
        </div>
      </div>
      <EvaluationTable
        evaluation_name={evaluation_name}
        selected_evaluation_run_infos={selected_evaluation_run_infos}
        evaluation_results={evaluation_results}
        evaluation_statistics={evaluation_statistics}
        evaluator_names={evaluator_names}
        selectedRows={selectedRows}
        setSelectedRows={setSelectedRows}
      />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= total_datapoints}
      />
      {!has_selected_runs && (
        <div className="mt-4 text-center text-gray-500">
          Select evaluation run IDs to view results
        </div>
      )}
    </>
  );
}

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const {
    evaluation_name,
    evaluationData,
    has_selected_runs,
    offset,
    limit,
    evaluator_names,
    running_evaluation_run_ids,
    errors,
    newFeedbackId,
    newJudgeDemonstrationId,
  } = loaderData;
  const location = useLocation();
  const isReadOnly = useReadOnly();
  const { toast } = useToast();
  const fetcher = useFetcher();
  const killFetcher = useFetcher();
  const isKilling = killFetcher.state !== "idle";
  const killSucceeded =
    (killFetcher.data as { success?: boolean } | undefined)?.success === true;
  const any_evaluation_is_running =
    running_evaluation_run_ids.length > 0 && !isKilling && !killSucceeded;

  const [selectedRows, setSelectedRows] = useState<
    Map<string, SelectedRowData>
  >(new Map());
  const [selectedDataset, setSelectedDataset] = useState<string>("");

  const config = useConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw data(
      `Evaluation config not found for evaluation ${evaluation_name}`,
      { status: 404 },
    );
  }
  const function_name = evaluation_config.function_name;

  useAutoRefresh(any_evaluation_is_running);

  const handleKillEvaluation = useCallback(() => {
    for (const runId of running_evaluation_run_ids) {
      killFetcher.submit(null, {
        method: "POST",
        action: `/api/evaluations/${encodeURIComponent(runId)}/kill`,
      });
    }
  }, [killFetcher, running_evaluation_run_ids]);

  useEffect(() => {
    if (killFetcher.state === "idle" && killFetcher.data) {
      const result = killFetcher.data as {
        success: boolean;
        error?: string;
      };
      if (!result.success && result.error) {
        toast.error({
          title: "Failed to stop evaluation",
          description: result.error,
        });
      }
    }
  }, [killFetcher.state, killFetcher.data, toast]);

  const hasErrorsToDisplay = Object.values(errors).some(
    (error) => error.errors.length > 0,
  );

  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, newJudgeDemonstrationId, toast]);

  useBulkAddToDatasetToast({
    fetcher,
    toast,
    setSelectedRows,
    setSelectedDataset,
  });

  const handleDatasetSelect = (dataset: string) => {
    setSelectedDataset(dataset);

    const selectedData = Array.from(selectedRows.values());

    if (selectedData.length === 0) {
      toast.error({ title: "No rows selected" });
      return;
    }

    const formData = new FormData();
    formData.append("_action", "addMultipleToDataset");
    formData.append("dataset", dataset);
    formData.append("evaluation_name", evaluation_name);
    formData.append("selectedItems", JSON.stringify(selectedData));

    fetcher.submit(formData, { method: "post" });
  };

  return (
    <PageLayout>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[{ label: "Evaluations", href: "/evaluations" }]}
          />
        }
        name={evaluation_name}
      >
        <BasicInfo evaluation_config={evaluation_config} />
        <ActionBar>
          <DatasetSelect
            selected={selectedDataset}
            onSelect={handleDatasetSelect}
            functionName={function_name}
            allowCreation
            disabled={isReadOnly || selectedRows.size === 0}
            placeholder={
              selectedRows.size > 0
                ? `Add ${selectedRows.size} selected ${selectedRows.size === 1 ? "inference" : "inferences"} to dataset`
                : "Add selected inferences to dataset"
            }
          />
          <AskAutopilotButton message={`Evaluation: ${evaluation_name}\n\n`} />
        </ActionBar>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          {hasErrorsToDisplay && (
            <>
              <SectionHeader heading="Errors" />
              <EvaluationErrorInfo errors={errors} />
            </>
          )}
          <Suspense key={location.key} fallback={<ResultsSkeleton />}>
            <Await resolve={evaluationData} errorElement={<ResultsError />}>
              {(resolvedData) => (
                <ResultsContent
                  evaluation_name={evaluation_name}
                  data={resolvedData}
                  evaluator_names={evaluator_names}
                  any_evaluation_is_running={any_evaluation_is_running}
                  has_selected_runs={has_selected_runs}
                  offset={offset}
                  limit={limit}
                  selectedRows={selectedRows}
                  setSelectedRows={setSelectedRows}
                  onKill={handleKillEvaluation}
                  isKilling={isKilling}
                />
              )}
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
