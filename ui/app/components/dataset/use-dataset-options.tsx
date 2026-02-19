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
 * Provides fetched dataset data, icons, and display helpers.
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

  const getPrefix = useCallback(
    (item: string | null, isSelected: boolean) => {
      if (!item) {
        return allowCreation ? (
          <TablePlus size={16} className="text-blue-600" />
        ) : (
          <Table
            size={16}
            className="text-fg-muted group-data-[selected=true]:text-menu-highlight-icon"
          />
        );
      }
      const exists = datasetsByName.has(item);
      if (isSelected && exists) {
        return <TableCheck size={16} className="text-green-700" />;
      }
      if (isSelected && !exists) {
        return <TablePlus size={16} className="text-blue-600" />;
      }
      return (
        <Table
          size={16}
          className="text-fg-muted group-data-[selected=true]:text-menu-highlight-icon"
        />
      );
    },
    [allowCreation, datasetsByName],
  );

  const getSuffix = useCallback(
    (item: string | null) => {
      if (!item) return null;
      const dataset = datasetsByName.get(item);
      if (!dataset) return null;
      return (
        <span className="bg-bg-tertiary text-fg-tertiary group-data-[selected=true]:bg-menu-highlight-badge group-data-[selected=true]:text-menu-highlight-badge-foreground shrink-0 rounded px-1.5 py-0.5 font-mono text-xs">
          {formatCompactNumber(dataset.count)}
        </span>
      );
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

  return {
    items: sortedDatasetNames,
    isLoading,
    isError,
    computedPlaceholder,
    searchPlaceholder,
    getPrefix,
    getSuffix,
    getSelectedDataset,
  };
}
