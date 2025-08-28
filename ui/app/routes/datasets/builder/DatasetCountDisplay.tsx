import { useDatasetInsertCountFetcher } from "~/routes/api/datasets/count_inserts.route";
import type { DatasetBuilderFormValues } from "./types";
import type { Control } from "react-hook-form";
import { Skeleton } from "~/components/ui/skeleton";
import { useEffect } from "react";

export function DatasetCountDisplay({
  control,
  setCountToInsert,
  onLoadingChange,
}: {
  control: Control<DatasetBuilderFormValues>;
  setCountToInsert: (count: number | null) => void;
  onLoadingChange?: (loading: boolean) => void;
}) {
  const { count, isLoading } = useDatasetInsertCountFetcher(control);
  // Notify parent of loading state
  useEffect(() => {
    onLoadingChange?.(isLoading);
  }, [isLoading, onLoadingChange]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2">
        There are currently <Skeleton className="inline-block h-4 w-16" /> rows
        to insert.
      </div>
    );
  }

  if (count === null) {
    return <div></div>;
  }

  setCountToInsert(count);

  return (
    <div>There are currently {count.toLocaleString()} rows to insert.</div>
  );
}
