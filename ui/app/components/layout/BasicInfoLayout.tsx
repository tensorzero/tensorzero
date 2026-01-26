import { Skeleton } from "~/components/ui/skeleton";

export function BasicInfoLayout({ children }: React.PropsWithChildren) {
  return <div className="flex flex-col gap-4 md:gap-2">{children}</div>;
}

export function BasicInfoItem({ children }: React.PropsWithChildren) {
  return <div className="flex flex-col gap-0.5 md:flex-row">{children}</div>;
}

export function BasicInfoItemTitle({ children }: React.PropsWithChildren) {
  return (
    <div className="text-fg-secondary w-full flex-shrink-0 text-left text-sm md:w-32 md:py-1">
      {children}
    </div>
  );
}

interface BasicInfoItemContentProps extends React.PropsWithChildren {
  wrap?: boolean;
}

export function BasicInfoItemContent({
  children,
  wrap = false,
}: BasicInfoItemContentProps) {
  return (
    <div
      className={`text-fg-primary flex flex-wrap gap-x-4 gap-y-0.5 md:gap-1 md:py-1 ${wrap ? "" : "truncate"}`}
    >
      {children}
    </div>
  );
}

interface BasicInfoLayoutSkeletonProps {
  rows?: number;
}

export function BasicInfoLayoutSkeleton({
  rows = 5,
}: BasicInfoLayoutSkeletonProps) {
  return (
    <BasicInfoLayout>
      {Array.from({ length: rows }).map((_, i) => (
        <BasicInfoItem key={i}>
          <BasicInfoItemTitle>
            <Skeleton className="h-5 w-20" />
          </BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Skeleton className="h-5 w-48" />
          </BasicInfoItemContent>
        </BasicInfoItem>
      ))}
    </BasicInfoLayout>
  );
}
