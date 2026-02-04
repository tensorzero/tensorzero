import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import type { Route } from "./+types/route";
import {
  Await,
  isRouteErrorResponse,
  useAsyncError,
  useLocation,
  useNavigate,
  useSearchParams,
  type RouteHandle,
} from "react-router";
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
import { Skeleton } from "~/components/ui/skeleton";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Runs",
    { label: match.params.run_id!, isIdentifier: true },
  ],
};

type RunData = {
  workflowEvaluationRun: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
    >
  >["runs"][0];
  workflowEvaluationRunEpisodes: Awaited<
    ReturnType<
      ReturnType<
        typeof getTensorZeroClient
      >["getWorkflowEvaluationRunEpisodesWithFeedback"]
    >
  >["episodes"];
  statistics: Awaited<
    ReturnType<
      ReturnType<
        typeof getTensorZeroClient
      >["getWorkflowEvaluationRunStatistics"]
    >
  >["statistics"];
  count: number;
};

async function fetchRunData(
  run_id: string,
  limit: number,
  offset: number,
): Promise<RunData> {
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

  const workflowEvaluationRuns = workflowEvaluationRunsResponse.runs;
  if (workflowEvaluationRuns.length !== 1) {
    throw new Error(`Workflow evaluation run "${run_id}" not found`);
  }

  return {
    workflowEvaluationRun: workflowEvaluationRuns[0],
    workflowEvaluationRunEpisodes:
      workflowEvaluationRunEpisodesResponse.episodes,
    statistics: statisticsResponse.statistics,
    count,
  };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const run_id = params.run_id;
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");

  return {
    run_id,
    runData: fetchRunData(run_id, limit, offset),
    offset,
    limit,
  };
}

function ContentSkeleton({ run_id }: { run_id: string }) {
  return (
    <>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Workflow Evaluations", href: "/workflow-evaluations" },
              { label: "Runs" },
            ]}
          />
        }
        name={run_id}
      />
      <SectionLayout>
        <Skeleton className="h-24 w-full" />
      </SectionLayout>
      <SectionLayout>
        <Skeleton className="h-64 w-full" />
      </SectionLayout>
    </>
  );
}

function ContentError({ run_id }: { run_id: string }) {
  const error = useAsyncError();
  let message = "Failed to load workflow evaluation run";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Workflow Evaluations", href: "/workflow-evaluations" },
              { label: "Runs" },
            ]}
          />
        }
        name={run_id}
      />
      <SectionErrorNotice
        icon={AlertCircle}
        title="Error loading workflow evaluation run"
        description={message}
      />
    </>
  );
}

function RunContent({
  data,
  offset,
  limit,
}: {
  data: RunData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const {
    workflowEvaluationRun,
    workflowEvaluationRunEpisodes,
    statistics,
    count,
  } = data;

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
    </>
  );
}

export default function WorkflowEvaluationRunSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const { run_id, runData, offset, limit } = loaderData;
  const location = useLocation();

  return (
    <PageLayout>
      <Suspense
        key={location.key}
        fallback={<ContentSkeleton run_id={run_id} />}
      >
        <Await
          resolve={runData}
          errorElement={<ContentError run_id={run_id} />}
        >
          {(data) => <RunContent data={data} offset={offset} limit={limit} />}
        </Await>
      </Suspense>
    </PageLayout>
  );
}
