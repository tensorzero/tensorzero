import { Skeleton } from "~/components/ui/skeleton";

export function SkeletonImage({
  className = "w-[150px]",
  error = undefined,
}: {
  className?: string;
  error?: string;
}) {
  if (error) {
    return (
      <Skeleton className={`relative aspect-square ${className}`}>
        <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
          <span className="text-center text-sm text-balance text-red-500/40">
            {error}
          </span>
        </div>
      </Skeleton>
    );
  } else {
    return (
      <Skeleton className={`relative aspect-square ${className}`}>
        <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
          <span className="text-muted-foreground/40">Image</span>
        </div>
      </Skeleton>
    );
  }
}
