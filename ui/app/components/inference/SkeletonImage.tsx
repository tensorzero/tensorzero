import { Skeleton } from "~/components/ui/skeleton";

export function SkeletonImage({
  className = "w-[150px]",
  error = false,
}: {
  className?: string;
  error?: boolean;
}) {
  if (error) {
    return (
      <Skeleton className={`relative aspect-square ${className}`}>
        <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
          <span className="text-balance text-center text-sm text-red-500/40">
            Failed to retrieve image.
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
