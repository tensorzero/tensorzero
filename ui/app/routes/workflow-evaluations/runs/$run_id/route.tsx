import type { Route } from "./+types/route";
import { data, useNavigate, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import BasicInfo from "./WorkflowEvaluationRunBasicInfo";
import WorkflowEvaluationRunEpisodesTable from "./WorkflowEvaluationRunEpisodesTable";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Runs",
    { label: match.params.run_id!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const run_id = params.run_id;
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");
  const tensorZeroClient = getTensorZeroClient();
  const [
    workflowEvaluationRunsResponse,
    workflowEvaluationRunEpisodesResponse,
    count,
    statisticsResponse,
  ] = await Promise.all([
    tensorZeroClient.listWorkflowEvaluationRuns(5, 0, run_id),
    tensorZeroClient.getWorkflowEvaluationRunEpisodesWithFeedback(
      run_id,
      limit,
      offset,
    ),
    tensorZeroClient.countWorkflowEvaluationRunEpisodes(run_id),
    tensorZeroClient.getWorkflowEvaluationRunStatistics(run_id),
  ]);
  const statistics = statisticsResponse.statistics;
  const workflowEvaluationRuns = workflowEvaluationRunsResponse.runs;
  if (workflowEvaluationRuns.length != 1) {
    throw data(`Workflow evaluation run "${run_id}" not found`, {
      status: 404,
    });
  }
  const workflowEvaluationRun = workflowEvaluationRuns[0];
  return {
    workflowEvaluationRun,
    workflowEvaluationRunEpisodes:
      workflowEvaluationRunEpisodesResponse.episodes,
    statistics,
    count,
    offset,
    limit,
  };
}

export default function WorkflowEvaluationRunSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const {
    workflowEvaluationRun,
    workflowEvaluationRunEpisodes,
    statistics,
    count,
    offset,
    limit,
  } = loaderData;

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
    <PageLayout>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Workflow Evaluations", href: "/workflow-evaluations" },
              { label: "Runs" },
            ]}
          />
        }
        name={workflowEvaluationRun.id}
      />
      <BasicInfo workflowEvaluationRun={workflowEvaluationRun} count={count} />
      <SectionLayout>
        <WorkflowEvaluationRunEpisodesTable
          episodes={workflowEvaluationRunEpisodes}
          statistics={statistics}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + limit >= count}
        />
      </SectionLayout>
    </PageLayout>
  );
}
