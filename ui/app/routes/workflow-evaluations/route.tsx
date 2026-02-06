import type { Route } from "./+types/route";
import { useLocation } from "react-router";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { fetchProjectsTableData, fetchRunsTableData } from "./route.server";
import { ProjectsSection } from "./ProjectsSection";
import { RunsSection } from "./RunsSection";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const runOffset = parseInt(searchParams.get("runOffset") || "0");
  const runLimit = parseInt(searchParams.get("runLimit") || "15");
  const projectOffset = parseInt(searchParams.get("projectOffset") || "0");
  const projectLimit = parseInt(searchParams.get("projectLimit") || "15");

  const client = getTensorZeroClient();
  const projectCountPromise = client.countWorkflowEvaluationProjects();
  const runCountPromise = client.countWorkflowEvaluationRuns();

  return {
    projectCountPromise,
    projectsData: fetchProjectsTableData(
      projectLimit,
      projectOffset,
      projectCountPromise,
    ),
    runCountPromise,
    runsData: fetchRunsTableData(runLimit, runOffset, runCountPromise),
    runOffset,
    runLimit,
    projectOffset,
    projectLimit,
  };
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    projectCountPromise,
    projectsData,
    runCountPromise,
    runsData,
    runOffset,
    runLimit,
    projectOffset,
    projectLimit,
  } = loaderData;
  const location = useLocation();

  return (
    <PageLayout>
      <PageHeader heading="Workflow Evaluations" />
      <ProjectsSection
        promise={projectsData}
        countPromise={projectCountPromise}
        offset={projectOffset}
        limit={projectLimit}
        locationKey={location.key}
      />
      <RunsSection
        promise={runsData}
        countPromise={runCountPromise}
        offset={runOffset}
        limit={runLimit}
        locationKey={location.key}
      />
    </PageLayout>
  );
}
