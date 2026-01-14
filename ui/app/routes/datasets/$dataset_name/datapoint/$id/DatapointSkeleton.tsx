import {
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import { Skeleton } from "~/components/ui/skeleton";

function PageHeaderSkeleton() {
  return (
    <div className="flex flex-col gap-2">
      <Skeleton className="h-4 w-16" />
      <Skeleton className="h-8 w-72" />
    </div>
  );
}

function BasicInfoSkeleton() {
  return (
    <BasicInfoLayout>
      {Array.from({ length: 5 }).map((_, i) => (
        <BasicInfoItem key={i}>
          <BasicInfoItemTitle>
            <Skeleton className="h-4 w-20" />
          </BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Skeleton className="h-6 w-32" />
          </BasicInfoItemContent>
        </BasicInfoItem>
      ))}
    </BasicInfoLayout>
  );
}

function ActionsSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-9 w-32" />
      <Skeleton className="h-9 w-32" />
      <Skeleton className="h-9 w-24" />
      <Skeleton className="h-9 w-20" />
    </div>
  );
}

function InputSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

function OutputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

function TagsSkeleton() {
  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Skeleton className="h-6 w-24" />
        <Skeleton className="h-6 w-32" />
      </div>
      <div className="flex gap-2">
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-6 w-28" />
      </div>
    </div>
  );
}

export function DatapointContentSkeleton() {
  return (
    <>
      <PageHeaderSkeleton />
      <SectionsGroup>
        <SectionLayout>
          <BasicInfoSkeleton />
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
