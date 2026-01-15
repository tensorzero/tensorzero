import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import { Skeleton } from "~/components/ui/skeleton";

function ActionsSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-24" />
      <Skeleton className="h-8 w-8" />
      <Skeleton className="h-8 w-8" />
    </div>
  );
}

function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

function OutputSkeleton() {
  return <Skeleton className="h-48 w-full" />;
}

function TagsSkeleton() {
  return <Skeleton className="h-16 w-full" />;
}

interface DatapointContentSkeletonProps {
  /** Datapoint ID from route params - shown immediately in header */
  id?: string;
}

export function DatapointContentSkeleton({
  id,
}: DatapointContentSkeletonProps) {
  return (
    <>
      <PageHeader label="Datapoint" name={id} />
      <SectionsGroup>
        <SectionLayout>
          <BasicInfoLayoutSkeleton rows={6} />
        </SectionLayout>

        <SectionLayout>
          <ActionsSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <OutputSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsSkeleton />
        </SectionLayout>
      </SectionsGroup>
    </>
  );
}
