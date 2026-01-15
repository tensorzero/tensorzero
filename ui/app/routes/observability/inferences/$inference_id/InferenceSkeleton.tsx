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
      {/* TryWithButton dropdown */}
      <Skeleton className="h-8 w-36" />
      {/* AddToDatasetButton */}
      <Skeleton className="h-8 w-36" />
      {/* HumanFeedbackButton */}
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

function FeedbackSkeleton() {
  return <Skeleton className="h-24 w-full" />;
}

function ParametersSkeleton() {
  return <Skeleton className="h-20 w-full" />;
}

function TagsSkeleton() {
  return <Skeleton className="h-16 w-full" />;
}

function ModelInferencesSkeleton() {
  return <Skeleton className="h-24 w-full" />;
}

interface InferenceContentSkeletonProps {
  /** Inference ID from route params - shown immediately in header */
  id?: string;
}

export function InferenceContentSkeleton({
  id,
}: InferenceContentSkeletonProps) {
  return (
    <>
      <PageHeader label="Inference" name={id}>
        <BasicInfoLayoutSkeleton rows={5} />
        <ActionsSkeleton />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <OutputSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Feedback" />
          <FeedbackSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Inference Parameters" />
          <ParametersSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsSkeleton />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Model Inferences" />
          <ModelInferencesSkeleton />
        </SectionLayout>
      </SectionsGroup>
    </>
  );
}
