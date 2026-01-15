import { Table, TablePlus } from "~/components/icons/Icons";
import { ButtonIcon } from "~/components/ui/button";
import { ButtonSelect } from "~/components/ui/select/ButtonSelect";
import {
  useDatasetOptions,
  getDatasetItemDataAttributes,
} from "./use-dataset-options";

interface DatasetSelectProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetSelect({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = false,
  disabled = false,
}: DatasetSelectProps) {
  const {
    items,
    isLoading,
    isError,
    searchPlaceholder,
    getPrefix,
    getSuffix,
    getSelectedDataset,
  } = useDatasetOptions({ functionName, allowCreation });

  const selectedDataset = getSelectedDataset(selected);

  const renderTrigger = () => {
    if (selected) {
      return (
        <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
          {selectedDataset ? (
            <Table size={16} className="shrink-0 text-green-700" />
          ) : (
            <TablePlus size={16} className="shrink-0 text-blue-600" />
          )}
          <span className="truncate font-mono text-sm">
            {selectedDataset?.name ?? selected}
          </span>
          <div className="ml-auto">{getSuffix(selected)}</div>
        </div>
      );
    }

    return (
      <span className="flex flex-row items-center gap-2">
        <ButtonIcon as={Table} variant="tertiary" />
        <span className="text-fg-primary flex text-sm font-medium">
          {placeholder}
        </span>
      </span>
    );
  };

  return (
    <ButtonSelect
      items={items}
      onSelect={onSelect}
      selected={selected}
      trigger={renderTrigger()}
      triggerClassName="group justify-between border"
      placeholder={searchPlaceholder}
      emptyMessage="No datasets found"
      disabled={disabled}
      isLoading={isLoading}
      loadingMessage="Loading datasets..."
      isError={isError}
      errorMessage="There was an error loading datasets."
      creatable={allowCreation}
      createHeading="New dataset"
      existingHeading="Existing"
      getPrefix={getPrefix}
      getSuffix={getSuffix}
      getItemDataAttributes={getDatasetItemDataAttributes}
    />
  );
}
