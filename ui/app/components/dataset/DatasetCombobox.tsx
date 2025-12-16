import { Combobox } from "~/components/ui/combobox";
import {
  useDatasetOptions,
  getDatasetItemDataAttributes,
} from "./use-dataset-options";

interface DatasetComboboxProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetCombobox({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = false,
  disabled = false,
}: DatasetComboboxProps) {
  const {
    items,
    isLoading,
    isError,
    computedPlaceholder,
    getPrefix,
    getSuffix,
  } = useDatasetOptions({ functionName, placeholder, allowCreation });

  return (
    <Combobox
      selected={selected}
      onSelect={onSelect}
      items={items}
      getPrefix={getPrefix}
      getSuffix={getSuffix}
      getItemDataAttributes={getDatasetItemDataAttributes}
      placeholder={computedPlaceholder}
      emptyMessage="No datasets found"
      disabled={disabled}
      allowCreation={allowCreation}
      creationHint={allowCreation ? "Type to create a new dataset" : undefined}
      createHeading={allowCreation ? "New dataset" : undefined}
      loading={isLoading}
      loadingMessage="Loading datasets..."
      error={isError}
      errorMessage="There was an error loading datasets."
    />
  );
}
