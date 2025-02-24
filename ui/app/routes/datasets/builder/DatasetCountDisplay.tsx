import { useDatasetCountFetcher } from "~/routes/api/datasets/count_inserts.route";
import type { DatasetBuilderFormValues } from "./types";
import type { Control } from "react-hook-form";

export function DatasetCountDisplay({
  control,
  setCountToInsert,
}: {
  control: Control<DatasetBuilderFormValues>;
  setCountToInsert: (count: number | null) => void;
}) {
  const { count, isLoading } = useDatasetCountFetcher(control);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (count === null) {
    return <div></div>;
  }

  setCountToInsert(count);

  return (
    <div>There are currently {count.toLocaleString()} rows to insert.</div>
  );
}
