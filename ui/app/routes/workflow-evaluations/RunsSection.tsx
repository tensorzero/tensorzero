import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import {
  Await,
  useAsyncError,
  useNavigate,
  useSearchParams,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import WorkflowEvaluationRunsTable from "./WorkflowEvaluationRunsTable";
import { Skeleton } from "~/components/ui/skeleton";
import {
  getErrorMessage,
  SectionErrorNotice,
} from "~/components/ui/error/ErrorContentPrimitives";
import type { RunsData } from "./route.server";

interface RunsSectionProps {
  promise: Promise<RunsData>;
  offset: number;
  limit: number;
  locationKey: string;
}

export function RunsSection({
  promise,
  offset,
  limit,
  locationKey,
}: RunsSectionProps) {
  return (
    <SectionLayout>
      <Suspense key={`runs-${locationKey}`} fallback={<RunsSkeleton />}>
        <Await resolve={promise} errorElement={<RunsError />}>
          {(data) => (
            <RunsContent data={data} offset={offset} limit={limit} />
          )}
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
  data: RunsData;
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
      <SectionHeader heading="Evaluation Runs" count={count} />
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
      <Skeleton className="mb-2 h-6 w-32" />
      <Skeleton className="h-48 w-full" />
    </>
  );
}

function RunsError() {
  const error = useAsyncError();
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading evaluation runs"
      description={getErrorMessage({
        error,
        fallback: "Failed to load evaluation runs",
      })}
    />
  );
}
