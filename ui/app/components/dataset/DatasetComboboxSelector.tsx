import { useMemo } from "react";
import { Combobox } from "~/components/ui/combobox";
import { useDatasetSelector } from "~/hooks/use-dataset-selector";

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
    sortedDatasetNames,
    isLoading,
    isError,
    getItemIcon,
    getItemSuffix,
    getItemDataAttributes,
  } = useDatasetSelector(functionName);

  const computedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [placeholder, allowCreation, sortedDatasetNames.length]);

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
