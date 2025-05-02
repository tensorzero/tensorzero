import { PageHeader, SectionLayout } from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { DynamicEvalRunSelector } from "~/components/dynamic_evaluations/DynamicEvalRunSelector";
import {
  getDynamicEvaluationRunsByIds,
  getDynamicEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/dynamic_evaluations.server";
import type { DynamicEvaluationRunStatisticsByMetricName } from "~/utils/clickhouse/dynamic_evaluations";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";

export async function loader({ request, params }: Route.LoaderArgs) {
  const projectName = params.project_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  //   const offset = parseInt(searchParams.get("offset") || "0");
  //   const pageSize = parseInt(searchParams.get("pageSize") || "15");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  const runStats: Record<string, DynamicEvaluationRunStatisticsByMetricName[]> =
    {};

  if (runIds.length > 0) {
    // Create promises for fetching statistics for each runId
    const statsPromises = runIds.map((runId) =>
      getDynamicEvaluationRunStatisticsByMetricName(runId, projectName),
    );

    // Create promise for fetching run info
    const runInfosPromise = getDynamicEvaluationRunsByIds(runIds, projectName);

    // Run all promises concurrently
    const [statsResults, runInfos] = await Promise.all([
      Promise.all(statsPromises), // Wait for all stats promises
      runInfosPromise, // Wait for run info promise
    ]);

    // Construct the runStats object from the results
    runIds.forEach((runId, index) => {
      runStats[runId] = statsResults[index];
    });

    return {
      projectName,
      runInfos,
      runStats, // Return runStats
    };
  } else {
    // Handle the case where there are no runIds
    return {
      projectName,
      runInfos: [],
      runStats: {}, // Return empty runStats
    };
  }
}

export default function DynamicEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, runInfos, runStats } = loaderData;
  const selectedRunIds = runInfos.map((run) => run.id);

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader
          heading={`Dynamic Evaluation Runs for Project ${projectName}`}
        />
        <SectionLayout>
          <div>
            <h1>Dynamic Evaluation Runs</h1>
          </div>
          <DynamicEvalRunSelector
            projectName={projectName}
            selectedRunInfos={runInfos}
          />
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
