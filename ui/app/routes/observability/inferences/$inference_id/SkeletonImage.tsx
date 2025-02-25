import { Skeleton } from "~/components/ui/skeleton";

export function SkeletonImage({
  className = "w-[150px]",
}: {
  className?: string;
}) {
  return (
    <Skeleton className={`relative aspect-square ${className}`}>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-muted-foreground/40">Image</span>
        <span className="text-sm text-muted-foreground/20">Coming soon...</span>
      </div>
    </Skeleton>
  );
}
