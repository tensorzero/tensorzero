import { ActionBar } from "~/components/layout/ActionBar";
import { VariantSelect } from "~/components/inference/VariantSelect";
import { EditButton } from "~/components/utils/EditButton";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { SaveButton } from "~/components/utils/SaveButton";
import { CancelButton } from "~/components/utils/CancelButton";
import { CloneDatapointButton } from "~/components/datapoint/CloneDatapointButton";
import { useReadOnly } from "~/context/read-only";
import type { Datapoint } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useConfig } from "~/context/config";

interface DatapointActionsProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  onModelSelect: (model: string) => void;
  variantInferenceIsLoading: boolean;
  onDelete: () => void;
  isDeleting: boolean;
  toggleEditing: () => void;
  onSave: () => void;
  canSave: boolean;
  isSaving: boolean;
  isEditing: boolean;
  onReset: () => void;
  function_name: string;
  isStale: boolean;
  datapoint: Datapoint;
}

export function DatapointActions({
  variants,
  onVariantSelect,
  onModelSelect,
  variantInferenceIsLoading,
  onDelete,
  isDeleting,
  toggleEditing,
  onSave,
  canSave,
  isSaving,
  isEditing,
  onReset,
  function_name,
  isStale,
  datapoint,
}: DatapointActionsProps) {
  const isReadOnly = useReadOnly();
  const handleCancel = () => {
    onReset();
    toggleEditing();
  };

  const config = useConfig();
  const modelsSet = new Set<string>([...variants, ...config.model_names]);
  const models = [...modelsSet].sort();

  const isDefault = function_name === DEFAULT_FUNCTION;
  const options = isDefault ? models : variants;

  return (
    <ActionBar>
      <VariantSelect
        options={options}
        onSelect={isDefault ? onModelSelect : onVariantSelect}
        isLoading={variantInferenceIsLoading}
        isDefaultFunction={isDefault}
      />
      {!isEditing && <CloneDatapointButton datapoint={datapoint} />}
      {isEditing ? (
        <>
          <CancelButton onClick={handleCancel} disabled={isSaving} />
          <SaveButton
            disabled={!canSave || isReadOnly}
            isLoading={isSaving}
            onClick={onSave}
          />
        </>
      ) : (
        <EditButton
          onClick={toggleEditing}
          disabled={isStale || isReadOnly}
          tooltip={
            isReadOnly
              ? "Editing is disabled in read-only mode"
              : isStale
                ? "You can't edit a stale datapoint."
                : "Edit"
          }
        />
      )}
      <DeleteButton
        onClick={onDelete}
        isLoading={isDeleting}
        disabled={isStale || isReadOnly}
        tooltip={
          isReadOnly
            ? "Deletion is disabled in read-only mode"
            : isStale
              ? "You can't delete a stale datapoint."
              : "Delete"
        }
      />
    </ActionBar>
  );
}
