import { Suspense } from "react";
import {
  Await,
  useNavigate,
  useSearchParams,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import WorkflowEvaluationRunsTable from "./WorkflowEvaluationRunsTable";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import type { RunsTableData } from "./route.server";
import type { CountValue } from "~/components/layout/CountDisplay";

interface RunsSectionProps {
  promise: Promise<RunsTableData>;
  countPromise: CountValue;
  offset: number;
  limit: number;
  locationKey: string;
}

export function RunsSection({
  promise,
  countPromise,
  offset,
  limit,
  locationKey,
}: RunsSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Evaluation Runs" count={countPromise} />
      <Suspense key={`runs-${locationKey}`} fallback={<RunsSkeleton />}>
        <Await
          resolve={promise}
          errorElement={<RunsError />}
        >
          {(data) => <RunsContent data={data} offset={offset} limit={limit} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function RunsContent({
  data,
  offset,
  limit,
}: {
  data: RunsTableData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { runs, count } = data;

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("runOffset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("runOffset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <WorkflowEvaluationRunsTable workflowEvaluationRuns={runs} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= count}
      />
    </>
  );
}

function RunsSkeleton() {
  return (
    <>
      <Skeleton className="h-48 w-full" />
      <PageButtons disabled />
    </>
  );
}

function RunsError() {
  return (
    <>
      <SectionAsyncErrorState defaultMessage="Failed to load evaluation runs" />
      <PageButtons disabled />
    </>
  );
}
