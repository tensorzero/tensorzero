import { useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";

// Skeleton
export function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

// Error
export function InputSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load input";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      {message}
    </div>
  );
}
