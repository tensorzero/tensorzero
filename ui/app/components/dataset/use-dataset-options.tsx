import { useCallback, useMemo } from "react";
import { Table, TablePlus, TableCheck } from "~/components/icons/Icons";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import { formatCompactNumber } from "~/utils/chart";

export function getDatasetItemDataAttributes(item: string) {
  return { "data-dataset-name": item };
}

interface UseDatasetOptionsParams {
  functionName?: string;
  placeholder?: string;
  allowCreation?: boolean;
}

/**
 * Shared hook for dataset selection components (DatasetCombobox, DatasetSelect).
 * Provides fetched dataset data, icons, filtering, and create-option logic.
 */
export function useDatasetOptions({
  functionName,
  placeholder,
  allowCreation = false,
}: UseDatasetOptionsParams) {
  const {
    data: datasets = [],
    isLoading,
    isError,
  } = useDatasetCounts(functionName);

  const sortedDatasetNames = useMemo(
    () =>
      [...datasets]
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

  const computedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [placeholder, allowCreation, sortedDatasetNames.length]);

  const searchPlaceholder = useMemo(() => {
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Search datasets";
  }, [allowCreation, sortedDatasetNames.length]);

  const getItemIcon = useCallback(
    (item: string | null, isSelected: boolean) => {
      if (!item) {
        return <TablePlus size={16} className="text-blue-600" />;
      }
      const exists = datasetsByName.has(item);
      if (isSelected && exists) {
        return <TableCheck size={16} className="text-green-700" />;
      }
      if (isSelected && !exists) {
        return <TablePlus size={16} className="text-blue-600" />;
      }
      return <Table size={16} className="text-fg-muted" />;
    },
    [datasetsByName],
  );

  const getItemSuffix = useCallback(
    (item: string | null) => {
      if (!item) return null;
      const dataset = datasetsByName.get(item);
      if (!dataset) return null;
      return formatCompactNumber(dataset.count);
    },
    [datasetsByName],
  );

  const getSelectedDataset = useCallback(
    (name: string | null) => {
      if (!name) return undefined;
      return datasetsByName.get(name);
    },
    [datasetsByName],
  );

  const filterItems = useCallback(
    (searchValue: string) => {
      const query = searchValue.toLowerCase();
      if (!query) return sortedDatasetNames;
      return sortedDatasetNames.filter((item) =>
        item.toLowerCase().includes(query),
      );
    },
    [sortedDatasetNames],
  );

  const shouldShowCreateOption = useCallback(
    (searchValue: string) => {
      return (
        allowCreation &&
        Boolean(searchValue.trim()) &&
        !sortedDatasetNames.some(
          (name) => name.toLowerCase() === searchValue.trim().toLowerCase(),
        )
      );
    },
    [allowCreation, sortedDatasetNames],
  );

  return {
    isLoading,
    isError,
    computedPlaceholder,
    searchPlaceholder,
    getItemIcon,
    getItemSuffix,
    getSelectedDataset,
    filterItems,
    shouldShowCreateOption,
  };
}
