import {
  PageHeader,
  SectionLayout,
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { WorkflowEvalRunSelector } from "~/routes/workflow-evaluations/projects/$project_name/WorkflowEvalRunSelector";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";
import { WorkflowEvaluationProjectResultsTable } from "./WorkflowEvaluationProjectResultsTable";
import { useNavigate, useSearchParams, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { fetchResultsData, type ResultsData } from "./route.server";

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
    const { runInfos, runStats, episodeInfo, count } = await fetchResultsData(
      runIds,
      projectName,
      limit,
      offset,
    );

    return {
      projectName,
      runInfos,
      runStats,
      episodeInfo,
      count,
      limit,
      offset,
    };
  } else {
    return {
      projectName,
      runInfos: [],
      runStats: {},
      episodeInfo: [],
      count: 0,
      limit,
      offset,
    };
  }
}

function ResultsContent({
  runInfos,
  runStats,
  episodeInfo,
  count,
  limit,
  offset,
}: {
  runInfos: ResultsData["runInfos"];
  runStats: ResultsData["runStats"];
  episodeInfo: ResultsData["episodeInfo"];
  count: number;
  limit: number;
  offset: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("offset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("offset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <WorkflowEvaluationProjectResultsTable
        selected_run_infos={runInfos}
        evaluation_results={episodeInfo}
        evaluation_statistics={runStats}
      />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= count}
      />
    </>
  );
}

export default function WorkflowEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, runInfos, runStats, episodeInfo, count, limit, offset } =
    loaderData;
  const selectedRunIds = runInfos.map((run) => run.id);

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
          <ResultsContent
            runInfos={runInfos}
            runStats={runStats}
            episodeInfo={episodeInfo}
            count={count}
            limit={limit}
            offset={offset}
          />
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
