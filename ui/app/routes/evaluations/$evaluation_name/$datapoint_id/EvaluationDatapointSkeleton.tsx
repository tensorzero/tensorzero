import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import { Skeleton } from "~/components/ui/skeleton";

function EvalRunSelectorSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      {/* Eval run selector dropdown */}
      <Skeleton className="h-8 w-48" />
    </div>
  );
}

function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

function OutputsSkeleton() {
  return (
    <div className="flex gap-4 overflow-x-auto">
      {/* At least 2 output columns */}
      <div className="min-w-64 flex-1 space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
      <div className="min-w-64 flex-1 space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    </div>
  );
}

interface EvaluationDatapointContentSkeletonProps {
  /** Datapoint ID from route params - shown immediately in header */
  datapointId?: string;
}

export function EvaluationDatapointContentSkeleton({
  datapointId,
}: EvaluationDatapointContentSkeletonProps) {
  return (
    <>
      <PageHeader label="Datapoint" name={datapointId}>
        <BasicInfoLayoutSkeleton rows={5} />
        <EvalRunSelectorSkeleton />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <OutputsSkeleton />
        </SectionLayout>
      </SectionsGroup>
    </>
  );
}
