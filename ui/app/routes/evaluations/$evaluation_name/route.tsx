import type { Route } from "./+types/route";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import {
  getEvaluationStatistics,
  getEvaluationResults,
  getEvaluationRunInfos,
  countDatapointsForEvaluation,
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
} from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import { data, redirect, useNavigate } from "react-router";
import AutoRefreshIndicator, { useAutoRefresh } from "./AutoRefreshIndicator";
import BasicInfo from "./EvaluationBasicInfo";
import { useConfig } from "~/context/config";
import { getRunningEvaluation } from "~/utils/evaluations.server";
import {
  EvaluationErrorInfo,
  type EvaluationErrorDisplayInfo,
} from "./EvaluationErrorInfo";
import { addEvaluationHumanFeedback } from "~/utils/tensorzero.server";
import { Toaster } from "~/components/ui/toaster";
import { useToast } from "~/hooks/use-toast";
import { useEffect, useState } from "react";
import { logger } from "~/utils/logger";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { ActionBar } from "~/components/layout/ActionBar";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { useFetcher, Link } from "react-router";
import { ToastAction } from "~/components/ui/toast";

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
  const pageSize = parseInt(searchParams.get("pageSize") || "15");

  const evaluator_names = Object.keys(evaluationConfig.evaluators);

  const metric_names = evaluator_names.map((evaluatorName) =>
    getEvaluatorMetricName(params.evaluation_name, evaluatorName),
  );

  // Set up all promises to run concurrently
  const evaluationRunInfosPromise = getEvaluationRunInfos(
    selected_evaluation_run_ids_array,
    function_name,
  );

  // Create placeholder promises for results and statistics that will be used conditionally
  let resultsPromise;
  if (selected_evaluation_run_ids_array.length > 0) {
    // If there is a freshly inserted feedback, ClickHouse may take some time to
    // update the evaluation results as it is eventually consistent.
    // In this case, we poll for the evaluation results until the feedback is found.
    resultsPromise = newFeedbackId
      ? pollForEvaluationResults(
          function_name,
          function_type,
          metric_names,
          selected_evaluation_run_ids_array,
          newFeedbackId,
          pageSize,
          offset,
        )
      : getEvaluationResults(
          function_name,
          function_type,
          metric_names,
          selected_evaluation_run_ids_array,
          pageSize,
          offset,
        );
  } else {
    resultsPromise = Promise.resolve([]);
  }

  let statisticsPromise;
  if (selected_evaluation_run_ids_array.length > 0) {
    statisticsPromise = getEvaluationStatistics(
      function_name,
      function_type,
      metric_names,
      selected_evaluation_run_ids_array,
    );
  } else {
    statisticsPromise = Promise.resolve([]);
  }

  let total_datapoints_promise;
  if (selected_evaluation_run_ids_array.length > 0) {
    total_datapoints_promise = countDatapointsForEvaluation(
      function_name,
      function_type,
      selected_evaluation_run_ids_array,
    );
  } else {
    total_datapoints_promise = Promise.resolve(0);
  }

  // Wait for all promises to complete concurrently
  const [
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    total_datapoints,
  ] = await Promise.all([
    evaluationRunInfosPromise,
    resultsPromise,
    statisticsPromise,
    total_datapoints_promise,
  ]);

  const any_evaluation_is_running = Object.values(
    selected_evaluation_run_ids_array,
  ).some((evaluationRunId) => {
    const runningEvaluation = getRunningEvaluation(evaluationRunId);
    if (!runningEvaluation) {
      return false;
    }
    if (runningEvaluation.completed) {
      // If the evaluation has completed and the completion time is at least
      // 5 seconds ago, return false
      if (runningEvaluation.completed.getTime() + 5000 < Date.now()) {
        return false;
      }
    }
    return true;
  });

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
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    has_selected_runs: selected_evaluation_run_ids_array.length > 0,
    offset,
    pageSize,
    total_datapoints,
    evaluator_names,
    any_evaluation_is_running,
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
        const config = await getConfig();
        const evaluation_config =
          config.evaluations[evaluation_name.toString()];

        if (!evaluation_config) {
          return data(
            {
              error: `Evaluation config not found for ${evaluation_name}`,
              success: false,
            },
            { status: 404 },
          );
        }

        const function_name = evaluation_config.function_name;
        const errors: string[] = [];
        let successCount = 0;

        // Process each selected item
        for (const item of selectedItems) {
          const itemFormData = new FormData();
          itemFormData.append("dataset", dataset.toString());
          itemFormData.append("output", "inherit");
          itemFormData.append("inference_id", item.inference_id);
          itemFormData.append("function_name", function_name);
          itemFormData.append("variant_name", item.variant_name);
          itemFormData.append("episode_id", item.episode_id || "");
          itemFormData.append("_action", "addToDataset");

          try {
            await handleAddToDatasetAction(itemFormData);
            successCount++;
          } catch (error) {
            logger.error(
              `Failed to add inference ${item.inference_id} to dataset:`,
              error,
            );
            errors.push(`Failed to add inference ${item.inference_id}`);
          }
        }

        if (errors.length > 0 && successCount === 0) {
          return data(
            {
              error: `Failed to add all inferences: ${errors.join(", ")}`,
              success: false,
            },
            { status: 400 },
          );
        }

        return data({
          success: true,
          count: successCount,
          dataset: dataset.toString(),
          errors: errors.length > 0 ? errors : undefined,
        });
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

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const {
    evaluation_name,
    selected_evaluation_run_infos,
    evaluation_results,
    evaluation_statistics,
    has_selected_runs,
    offset,
    pageSize,
    total_datapoints,
    evaluator_names,
    any_evaluation_is_running,
    errors,
    newFeedbackId,
    newJudgeDemonstrationId,
  } = loaderData;
  const navigate = useNavigate();
  const { toast } = useToast();
  const fetcher = useFetcher();

  // State for tracking selected rows
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

  // Use that time for auto-refreshing
  useAutoRefresh(any_evaluation_is_running);

  const hasErrorsToDisplay = Object.values(errors).some(
    (error) => error.errors.length > 0,
  );

  // Handle feedback toast
  useEffect(() => {
    if (newFeedbackId) {
      toast({
        title: "Feedback Added",
      });
    }
  }, [newFeedbackId, newJudgeDemonstrationId, toast]);

  // Handle fetcher response for bulk add to dataset
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.error) {
        toast({
          title: "Failed to add to dataset",
          description: fetcher.data.error,
          variant: "destructive",
        });
      } else if (fetcher.data.success) {
        const datasetName = fetcher.data.dataset;
        toast({
          title: "Added to Dataset",
          description: `${fetcher.data.count} ${fetcher.data.count === 1 ? "inference" : "inferences"} added to: ${datasetName}`,
          action: (
            <ToastAction altText="View Dataset" asChild>
              <Link to={`/datasets/${datasetName}`}>View Dataset</Link>
            </ToastAction>
          ),
        });
        setSelectedRows(new Map());
        setSelectedDataset("");
      }
    }
  }, [fetcher.state, fetcher.data, toast, selectedDataset]);

  // Handle dataset selection for bulk add
  const handleDatasetSelect = (dataset: string) => {
    setSelectedDataset(dataset);

    const selectedData = Array.from(selectedRows.values());

    if (selectedData.length === 0) {
      toast({
        title: "No rows selected",
        variant: "destructive",
      });
      return;
    }

    // Submit the form with all selected items
    const formData = new FormData();
    formData.append("_action", "addMultipleToDataset");
    formData.append("dataset", dataset);
    formData.append("evaluation_name", evaluation_name);
    formData.append("selectedItems", JSON.stringify(selectedData));

    fetcher.submit(formData, { method: "post" });
  };

  return (
    <PageLayout>
      <PageHeader label="Evaluation" name={evaluation_name}>
        <BasicInfo evaluation_config={evaluation_config} />
        <ActionBar>
          <DatasetSelector
            selected={selectedDataset}
            onSelect={handleDatasetSelect}
            functionName={function_name}
            placeholder={
              selectedRows.size > 0
                ? `Add ${selectedRows.size} selected ${selectedRows.size === 1 ? "inference" : "inferences"} to dataset`
                : "Add selected inferences to dataset"
            }
            buttonProps={{
              size: "sm",
            }}
            disabled={selectedRows.size === 0}
          />
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
          <div className="flex items-center">
            <SectionHeader heading="Results" />
            <div
              className="ml-4"
              data-testid="auto-refresh-wrapper"
              data-running={any_evaluation_is_running}
            >
              <AutoRefreshIndicator isActive={any_evaluation_is_running} />
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
            disableNext={offset + pageSize >= total_datapoints}
          />
          {!has_selected_runs && (
            <div className="mt-4 text-center text-gray-500">
              Select evaluation run IDs to view results
            </div>
          )}
        </SectionLayout>
      </SectionsGroup>
      <Toaster />
    </PageLayout>
  );
}
