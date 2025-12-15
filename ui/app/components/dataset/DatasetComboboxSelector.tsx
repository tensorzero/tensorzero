import { useCallback, useMemo } from "react";
import { Table, TablePlus, TableCheck } from "~/components/icons/Icons";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import { Combobox } from "~/components/ui/combobox";

interface DatasetComboboxSelectorProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetComboboxSelector({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = false,
  disabled = false,
}: DatasetComboboxSelectorProps) {
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

  const computedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [placeholder, allowCreation, sortedDatasetNames.length]);

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

  return (
    <Combobox
      selected={selected}
      onSelect={onSelect}
      items={sortedDatasetNames}
      getItemIcon={getItemIcon}
      getItemSuffix={getItemSuffix}
      getItemDataAttributes={getItemDataAttributes}
      placeholder={computedPlaceholder}
      emptyMessage="No datasets found."
      disabled={disabled}
      allowCreation={allowCreation}
      creationHint={allowCreation ? "Type to create a new dataset" : undefined}
      createHeading="Create dataset"
      loading={isLoading}
      loadingMessage="Loading datasets..."
      error={isError}
      errorMessage="There was an error loading datasets."
    />
  );
}
