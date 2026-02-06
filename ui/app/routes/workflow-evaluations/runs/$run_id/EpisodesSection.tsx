import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import {
  Await,
  useAsyncError,
  useNavigate,
  useSearchParams,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { SectionLayout } from "~/components/layout/PageLayout";
import WorkflowEvaluationRunEpisodesTable from "./WorkflowEvaluationRunEpisodesTable";
import { Skeleton } from "~/components/ui/skeleton";
import {
  getErrorMessage,
  SectionErrorNotice,
} from "~/components/ui/error/ErrorContentPrimitives";
import type { EpisodesData } from "./route.server";

interface EpisodesSectionProps {
  promise: Promise<EpisodesData>;
  offset: number;
  limit: number;
  locationKey: string;
}

export function EpisodesSection({
  promise,
  offset,
  limit,
  locationKey,
}: EpisodesSectionProps) {
  return (
    <SectionLayout>
      <Suspense key={`episodes-${locationKey}`} fallback={<EpisodesSkeleton />}>
        <Await resolve={promise} errorElement={<EpisodesError />}>
          {(data) => (
            <EpisodesContent data={data} offset={offset} limit={limit} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
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
  const [searchParams] = useSearchParams();
  const { episodes, statistics, count } = data;

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

function EpisodesSkeleton() {
  return (
    <>
      <Skeleton className="h-64 w-full" />
      <PageButtons disabled />
    </>
  );
}

function EpisodesError() {
  const error = useAsyncError();
  return (
    <>
      <SectionErrorNotice
        icon={AlertCircle}
        title="Error loading episodes"
        description={getErrorMessage({
          error,
          fallback: "Failed to load episodes",
        })}
      />
      <PageButtons disabled />
    </>
  );
}
