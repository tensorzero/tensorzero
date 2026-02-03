import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import type { Route } from "./+types/route";
import {
  Await,
  data,
  isRouteErrorResponse,
  useAsyncError,
  useLocation,
  useNavigate,
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

type BasicInfoData = {
  workflowEvaluationRun: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
    >
  >["runs"][0];
  count: number;
};

type EpisodesData = {
  episodes: Awaited<
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

async function fetchBasicInfoData(run_id: string): Promise<BasicInfoData> {
  const client = getTensorZeroClient();
  const [runsResponse, count] = await Promise.all([
    client.listWorkflowEvaluationRuns(5, 0, run_id),
    client.countWorkflowEvaluationRunEpisodes(run_id),
  ]);

  const runs = runsResponse.runs;
  if (runs.length !== 1) {
    throw data(`Workflow evaluation run "${run_id}" not found`, {
      status: 404,
    });
  }

  return {
    workflowEvaluationRun: runs[0],
    count,
  };
}

async function fetchEpisodesData(
  run_id: string,
  limit: number,
  offset: number,
): Promise<EpisodesData> {
  const client = getTensorZeroClient();
  const [episodesResponse, statisticsResponse, count] = await Promise.all([
    client.getWorkflowEvaluationRunEpisodesWithFeedback(run_id, limit, offset),
    client.getWorkflowEvaluationRunStatistics(run_id),
    client.countWorkflowEvaluationRunEpisodes(run_id),
  ]);

  return {
    episodes: episodesResponse.episodes,
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
    basicInfoData: fetchBasicInfoData(run_id),
    episodesData: fetchEpisodesData(run_id, limit, offset),
    offset,
    limit,
  };
}

function BasicInfoSkeleton() {
  return <Skeleton className="h-24 w-full" />;
}

function BasicInfoError() {
  const error = useAsyncError();
  let message = "Failed to load run info";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading run info"
      description={message}
    />
  );
}

function EpisodesSkeleton() {
  return <Skeleton className="h-64 w-full" />;
}

function EpisodesError() {
  const error = useAsyncError();
  let message = "Failed to load episodes";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading episodes"
      description={message}
    />
  );
}

function BasicInfoContent({ data }: { data: BasicInfoData }) {
  return (
    <BasicInfo
      workflowEvaluationRun={data.workflowEvaluationRun}
      count={data.count}
    />
  );
}

function EpisodesContent({
  data,
  offset,
  limit,
}: {
  data: EpisodesData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const { episodes, statistics, count } = data;

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
      <WorkflowEvaluationRunEpisodesTable
        episodes={episodes}
        statistics={statistics}
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

export default function WorkflowEvaluationRunSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const { run_id, basicInfoData, episodesData, offset, limit } = loaderData;
  const location = useLocation();

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
        name={run_id}
      />
      <Suspense
        key={`basic-info-${location.key}`}
        fallback={<BasicInfoSkeleton />}
      >
        <Await resolve={basicInfoData} errorElement={<BasicInfoError />}>
          {(data) => <BasicInfoContent data={data} />}
        </Await>
      </Suspense>
      <SectionLayout>
        <Suspense
          key={`episodes-${location.key}`}
          fallback={<EpisodesSkeleton />}
        >
          <Await resolve={episodesData} errorElement={<EpisodesError />}>
            {(data) => (
              <EpisodesContent data={data} offset={offset} limit={limit} />
            )}
          </Await>
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}
