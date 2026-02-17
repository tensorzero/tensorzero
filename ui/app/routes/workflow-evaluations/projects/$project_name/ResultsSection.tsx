import { Suspense } from "react";
import { Await, useNavigate, useSearchParams } from "react-router";
import type { WorkflowEvaluationRun } from "~/types/tensorzero";
import PageButtons from "~/components/utils/PageButtons";
import { WorkflowEvaluationProjectResultsTable } from "./WorkflowEvaluationProjectResultsTable";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import type { ResultsData } from "./route.server";

interface ResultsSectionProps {
  resultsData: Promise<ResultsData>;
  runInfos: WorkflowEvaluationRun[];
  limit: number;
  offset: number;
  locationKey: string;
}

export function ResultsSection({
  resultsData,
  runInfos,
  limit,
  offset,
  locationKey,
}: ResultsSectionProps) {
  return (
    <Suspense key={locationKey} fallback={<ResultsSkeleton />}>
      <Await resolve={resultsData} errorElement={<ResultsError />}>
        {(data) => (
          <ResultsContent
            runInfos={runInfos}
            data={data}
            limit={limit}
            offset={offset}
          />
        )}
      </Await>
    </Suspense>
  );
}

function ResultsContent({
  runInfos,
  data,
  limit,
  offset,
}: {
  runInfos: WorkflowEvaluationRun[];
  data: ResultsData;
  limit: number;
  offset: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { runStats, episodeInfo, count } = data;

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

function ResultsSkeleton() {
  return (
    <>
      <Skeleton className="h-64 w-full" />
      <PageButtons disabled />
    </>
  );
}

function ResultsError() {
  return (
    <>
      <SectionAsyncErrorState defaultMessage="Failed to load results" />
      <PageButtons disabled />
    </>
  );
}
