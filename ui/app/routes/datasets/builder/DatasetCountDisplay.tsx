import { useDatasetInsertCountFetcher } from "~/routes/api/datasets/count_inserts.route";
import type { DatasetBuilderFormValues } from "./types";
import type { Control } from "react-hook-form";
import { Skeleton } from "~/components/ui/skeleton";

export function DatasetCountDisplay({
  control,
  setCountToInsert,
}: {
  control: Control<DatasetBuilderFormValues>;
  setCountToInsert: (count: number | null) => void;
}) {
  const { count, isLoading } = useDatasetInsertCountFetcher(control);

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
