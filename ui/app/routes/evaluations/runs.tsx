import { Suspense, useEffect, useState } from "react";
import { AlertCircle, Loader2, StopCircle } from "lucide-react";
import { Button } from "~/components/ui/button";
import type { Route } from "./+types/runs";
import type { EvaluationResultRow } from "~/types/tensorzero";
import {
  getEvaluationResults,
  pollForEvaluationResults,
} from "~/utils/clickhouse/evaluations.server";
import {
  EvaluationTable,
  type SelectedRowData,
} from "~/routes/evaluations/EvaluationTable";
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
import AutoRefreshIndicator, {
  useAutoRefresh,
} from "~/routes/evaluations/AutoRefreshIndicator";
import { getRunningEvaluation } from "~/utils/evaluations.server";
import {
  EvaluationErrorInfo,
  type EvaluationErrorDisplayInfo,
} from "~/routes/evaluations/EvaluationErrorInfo";
import {
  addEvaluationHumanFeedback,
  getTensorZeroClient,
} from "~/utils/tensorzero.server";
import { useToast } from "~/hooks/use-toast";
import { logger } from "~/utils/logger";
import { ActionBar } from "~/components/layout/ActionBar";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { DatasetSelect } from "~/components/dataset/DatasetSelect";
import { handleBulkAddToDataset } from "~/routes/evaluations/bulkAddToDataset.server";
import { useBulkAddToDatasetToast } from "~/routes/evaluations/useBulkAddToDatasetToast";
import { useCancelEvaluation } from "~/routes/evaluations/useCancelEvaluation";
import { useReadOnly } from "~/context/read-only";
import { SnapshotBanner } from "~/components/layout/SnapshotBanner";
import BasicInfo from "~/routes/evaluations/EvaluationBasicInfo";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import type { RunMetricMetadata } from "~/types/tensorzero";
import { mergeRunMetrics } from "~/utils/clickhouse/evaluations";

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

  const resultsPromise: Promise<EvaluationResultRow[]> = newFeedbackId
    ? pollForEvaluationResults(
        selected_evaluation_run_ids_array,
        newFeedbackId,
        limit,
        offset,
      )
    : getEvaluationResults(selected_evaluation_run_ids_array, limit, offset);

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

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);

  const runIdsParam = searchParams.get("evaluation_run_ids");
  if (!runIdsParam) {
    throw data("Missing `evaluation_run_ids` query parameter", { status: 400 });
  }

  const selected_evaluation_run_ids_array = runIdsParam
    .split(",")
    .filter((s) => s.trim().length > 0);
  if (selected_evaluation_run_ids_array.length === 0) {
    throw data("No run IDs provided", { status: 400 });
  }

  // Fetch run metadata from the database for all selected runs
  const client = getTensorZeroClient();
  const runMetadataResponse = await client.getEvaluationRunMetadata(
    selected_evaluation_run_ids_array,
  );

  // Merge metadata from all available runs
  const allMetadata = Object.values(runMetadataResponse.metadata);
  if (allMetadata.length === 0) {
    throw data(
      `Evaluation run metadata not found for any of the provided run IDs`,
      { status: 404 },
    );
  }

  const functionNames = new Set(allMetadata.map((m) => m.function_name));
  if (functionNames.size > 1) {
    throw data(
      `All evaluation runs must target the same function, but found: ${[...functionNames].join(", ")}`,
      { status: 400 },
    );
  }

  // Use the first selected run ID (not Object.values order) for primary
  // metadata, since the server returns a HashMap with no stable ordering.
  const primaryMetadata =
    runMetadataResponse.metadata[selected_evaluation_run_ids_array[0]] ??
    allMetadata[0];
  const evaluation_name = primaryMetadata.evaluation_name;
  const function_name = primaryMetadata.function_name;
  const function_type = primaryMetadata.function_type as "chat" | "json";

  const {
    metrics: mergedMetrics,
    metricsConfig,
    evaluatorMetricNames,
  } = mergeRunMetrics(allMetadata);
  const metric_names = mergedMetrics.map((m) => m.name);

  const newFeedbackId = searchParams.get("newFeedbackId");
  const newJudgeDemonstrationId = searchParams.get("newJudgeDemonstrationId");
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");

  const running_evaluation_run_ids = selected_evaluation_run_ids_array.filter(
    (evaluationRunId) => {
      const runningEvaluation = getRunningEvaluation(evaluationRunId);
      if (!runningEvaluation) {
        return false;
      }
      if (runningEvaluation.cancelled) {
        return false;
      }
      if (runningEvaluation.completed) {
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
    evaluation_name,
    function_name,
    snapshotHash: undefined as string | undefined,
    function_type,
    metricsConfig,
    evaluationData: fetchEvaluationData(
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
    metric_names,
    evaluatorMetricNames,
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
  metricsConfig,
  evaluatorMetricNames,
  data,
  metric_names,
  any_evaluation_is_running,
  has_selected_runs,
  offset,
  limit,
  selectedRows,
  setSelectedRows,
  onCancel,
  isCancelling,
}: {
  evaluation_name: string;
  metricsConfig: Record<string, RunMetricMetadata>;
  evaluatorMetricNames: Record<string, string>;
  data: EvaluationData;
  metric_names: string[];
  any_evaluation_is_running: boolean;
  has_selected_runs: boolean;
  offset: number;
  limit: number;
  selectedRows: Map<string, SelectedRowData>;
  setSelectedRows: React.Dispatch<
    React.SetStateAction<Map<string, SelectedRowData>>
  >;
  onCancel: () => void;
  isCancelling: boolean;
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
          {(any_evaluation_is_running || isCancelling) && (
            <Button
              variant="outline"
              size="sm"
              onClick={onCancel}
              disabled={isCancelling}
              slotLeft={
                isCancelling ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <StopCircle className="h-3.5 w-3.5" />
                )
              }
            >
              {isCancelling ? "Stopping..." : "Stop"}
            </Button>
          )}
        </div>
      </div>
      <EvaluationTable
        evaluation_name={evaluation_name}
        metricsConfig={metricsConfig}
        evaluatorMetricNames={evaluatorMetricNames}
        selected_evaluation_run_infos={selected_evaluation_run_infos}
        evaluation_results={evaluation_results}
        evaluation_statistics={evaluation_statistics}
        metric_names={metric_names}
        selectedRows={selectedRows}
        setSelectedRows={setSelectedRows}
      />
      {has_selected_runs ? (
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + limit >= total_datapoints}
        />
      ) : (
        <div className="mt-4 text-center text-gray-500">
          Select evaluation run IDs to view results
        </div>
      )}
    </>
  );
}

export default function EvaluationRunsPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    evaluation_name,
    function_name,
    snapshotHash,
    function_type,
    metricsConfig,
    evaluationData,
    has_selected_runs,
    offset,
    limit,
    metric_names,
    evaluatorMetricNames,
    running_evaluation_run_ids,
    errors,
    newFeedbackId,
    newJudgeDemonstrationId,
  } = loaderData;
  const location = useLocation();
  const isReadOnly = useReadOnly();
  const { toast } = useToast();
  const fetcher = useFetcher();
  const { isCancelling, anyEvaluationIsRunning, handleCancelEvaluation } =
    useCancelEvaluation({
      runningEvaluationRunIds: running_evaluation_run_ids,
    });

  const [selectedRows, setSelectedRows] = useState<
    Map<string, SelectedRowData>
  >(new Map());
  const [selectedDataset, setSelectedDataset] = useState<string>("");

  useAutoRefresh(anyEvaluationIsRunning);

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
        banner={snapshotHash ? <SnapshotBanner /> : undefined}
        eyebrow={
          <Breadcrumbs
            segments={[{ label: "Evaluations", href: "/evaluations" }]}
          />
        }
        name={evaluation_name}
      >
        <BasicInfo
          functionName={function_name}
          functionType={function_type}
          snapshotHash={snapshotHash}
        />
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
              {(resolvedData) => {
                return (
                  <ResultsContent
                    evaluation_name={evaluation_name}
                    metricsConfig={metricsConfig}
                    evaluatorMetricNames={evaluatorMetricNames}
                    data={resolvedData}
                    metric_names={metric_names}
                    any_evaluation_is_running={anyEvaluationIsRunning}
                    has_selected_runs={has_selected_runs}
                    offset={offset}
                    limit={limit}
                    selectedRows={selectedRows}
                    setSelectedRows={setSelectedRows}
                    onCancel={handleCancelEvaluation}
                    isCancelling={isCancelling}
                  />
                );
              }}
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
