import { useCallback, useMemo } from "react";
import { Table, TablePlus, TableCheck } from "~/components/icons/Icons";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";

export function useDatasetSelector(functionName?: string) {
  const {
    data: datasets = [],
    isLoading,
    isError,
  } = useDatasetCounts(functionName);

  const sortedDatasetNames = useMemo(
    () =>
      [...(datasets ?? [])]
        .sort(
          (a, b) =>
            new Date(b.lastUpdated).getTime() -
            new Date(a.lastUpdated).getTime(),
        )
        .map((d) => d.name),
    [datasets],
  );

  const datasetsByName = useMemo(
    () => new Map(datasets.map((d) => [d.name, d])),
    [datasets],
  );

  const getItemIcon = useCallback(
    (item: string | null, isSelected: boolean) => {
      if (!item) {
        // Creating new dataset
        return <TablePlus className="h-4 w-4 text-blue-600" />;
      }
      const exists = datasetsByName.has(item);
      if (isSelected && exists) {
        return <TableCheck size={16} className="text-green-700" />;
      }
      if (isSelected && !exists) {
        // Selected but doesn't exist yet (new dataset)
        return <TablePlus className="h-4 w-4 text-blue-600" />;
      }
      return <Table size={16} className="text-fg-muted" />;
    },
    [datasetsByName],
  );

  const getItemSuffix = useCallback(
    (item: string | null) => {
      if (!item) return null;
      const dataset = datasetsByName.get(item);
      return dataset?.count.toLocaleString();
    },
    [datasetsByName],
  );

  const getItemDataAttributes = useCallback(
    (item: string) => ({ "data-dataset-name": item }),
    [],
  );

  const getSelectedDataset = useCallback(
    (name: string | null | undefined) => {
      if (!name) return null;
      return datasetsByName.get(name);
    },
    [datasetsByName],
  );

  return {
    sortedDatasetNames,
    datasetsByName,
    isLoading,
    isError,
    getItemIcon,
    getItemSuffix,
    getItemDataAttributes,
    getSelectedDataset,
  };
}
