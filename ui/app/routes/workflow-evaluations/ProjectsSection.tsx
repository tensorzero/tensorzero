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
import WorkflowEvaluationProjectsTable from "./WorkflowEvaluationProjectsTable";
import { Skeleton } from "~/components/ui/skeleton";
import {
  getErrorMessage,
  SectionErrorNotice,
} from "~/components/ui/error/ErrorContentPrimitives";
import type { ProjectsData } from "./route.server";

interface ProjectsSectionProps {
  promise: Promise<ProjectsData>;
  offset: number;
  limit: number;
  locationKey: string;
}

export function ProjectsSection({
  promise,
  offset,
  limit,
  locationKey,
}: ProjectsSectionProps) {
  return (
    <SectionLayout>
      <Suspense key={`projects-${locationKey}`} fallback={<ProjectsSkeleton />}>
        <Await resolve={promise} errorElement={<ProjectsError />}>
          {(data) => (
            <ProjectsContent data={data} offset={offset} limit={limit} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function ProjectsContent({
  data,
  offset,
  limit,
}: {
  data: ProjectsData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { projects, count } = data;

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("projectOffset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("projectOffset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <SectionHeader heading="Projects" count={count} />
      <WorkflowEvaluationProjectsTable workflowEvaluationProjects={projects} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= count}
      />
    </>
  );
}

function ProjectsSkeleton() {
  return (
    <>
      <Skeleton className="mb-2 h-6 w-32" />
      <Skeleton className="h-48 w-full" />
      <PageButtons disabled />
    </>
  );
}

function ProjectsError() {
  const error = useAsyncError();
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading projects"
      description={getErrorMessage({
        error,
        fallback: "Failed to load projects",
      })}
    />
  );
}
