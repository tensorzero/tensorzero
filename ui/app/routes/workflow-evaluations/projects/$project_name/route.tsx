import {
  PageHeader,
  SectionLayout,
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { WorkflowEvalRunSelector } from "~/routes/workflow-evaluations/projects/$project_name/WorkflowEvalRunSelector";
import type { WorkflowEvaluationRun } from "~/types/tensorzero";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";
import { useLocation, type RouteHandle } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { fetchResultsData } from "./route.server";
import { ResultsSection } from "./ResultsSection";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Projects",
    { label: match.params.project_name!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const projectName = params.project_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const limit = parseInt(searchParams.get("limit") || "15");
  const offset = parseInt(searchParams.get("offset") || "0");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  if (runIds.length > 0) {
    // Start results fetch immediately so it runs concurrently with runInfos
    const resultsData = fetchResultsData(runIds, limit, offset);

    // Await runInfos synchronously — the run selector needs this data
    // and must remain outside the Suspense boundary
    const client = getTensorZeroClient();
    const runInfos = await client
      .getWorkflowEvaluationRuns(runIds, projectName)
      .then((response) => response.runs);
    runInfos.sort((a, b) => runIds.indexOf(a.id) - runIds.indexOf(b.id));

    return {
      projectName,
      runInfos,
      resultsData,
      limit,
      offset,
    };
  } else {
    return {
      projectName,
      runInfos: [] as WorkflowEvaluationRun[],
      resultsData: null,
      limit,
      offset,
    };
  }
}

export default function WorkflowEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, runInfos, resultsData, limit, offset } = loaderData;
  const selectedRunIds = runInfos.map((run) => run.id);
  const location = useLocation();

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader
          eyebrow={
            <Breadcrumbs
              segments={[
                {
                  label: "Workflow Evaluations",
                  href: "/workflow-evaluations",
                },
                { label: "Projects" },
              ]}
            />
          }
          name={projectName}
        />
        <SectionLayout>
          <WorkflowEvalRunSelector
            projectName={projectName}
            selectedRunInfos={runInfos}
          />
          {resultsData ? (
            <ResultsSection
              promise={resultsData}
              runInfos={runInfos}
              limit={limit}
              offset={offset}
              locationKey={location.key}
            />
          ) : null}
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
