import type { Route } from "./+types/route";
import { useLocation, type RouteHandle } from "react-router";
import {
  PageHeader,
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { fetchRunRecord, fetchEpisodesTableData } from "./route.server";
import { BasicInfoSection } from "./BasicInfoSection";
import { EpisodesSection } from "./EpisodesSection";
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

  const client = getTensorZeroClient();
  const countPromise = client.countWorkflowEvaluationRunEpisodes(run_id);
  const runRecordPromise = fetchRunRecord(run_id);
  const episodesTablePromise = fetchEpisodesTableData(run_id, limit, offset);

  return {
    run_id,
    basicInfoData: Promise.all([runRecordPromise, countPromise]).then(
      ([workflowEvaluationRun, count]) => ({ workflowEvaluationRun, count }),
    ),
    episodesData: Promise.all([episodesTablePromise, countPromise]).then(
      ([tableData, count]) => ({ ...tableData, count }),
    ),
    offset,
    limit,
  };
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
      <BasicInfoSection promise={basicInfoData} locationKey={location.key} />
      <EpisodesSection
        promise={episodesData}
        offset={offset}
        limit={limit}
        locationKey={location.key}
      />
    </PageLayout>
  );
}
