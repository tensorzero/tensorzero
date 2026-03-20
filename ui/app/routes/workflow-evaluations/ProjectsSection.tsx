import { Suspense } from "react";
import { Await, useNavigate, useSearchParams } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import WorkflowEvaluationProjectsTable from "./WorkflowEvaluationProjectsTable";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import type { ProjectsTableData } from "./route.server";
import type { CountValue } from "~/components/layout/CountDisplay";

interface ProjectsSectionProps {
  projectsData: Promise<ProjectsTableData>;
  countPromise: CountValue;
  offset: number;
  limit: number;
  locationKey: string;
}

export function ProjectsSection({
  projectsData,
  countPromise,
  offset,
  limit,
  locationKey,
}: ProjectsSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Projects" count={countPromise} />
      <Suspense key={`projects-${locationKey}`} fallback={<ProjectsSkeleton />}>
        <Await resolve={projectsData} errorElement={<ProjectsError />}>
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
  data: ProjectsTableData;
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
      <Skeleton className="h-48 w-full" />
      <PageButtons disabled />
    </>
  );
}

function ProjectsError() {
  return (
    <>
      <SectionAsyncErrorState defaultMessage="Failed to load projects" />
      <PageButtons disabled />
    </>
  );
}
