import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithButton } from "~/components/inference/TryWithButton";
import { EditButton } from "~/components/utils/EditButton";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { SaveButton } from "~/components/utils/SaveButton";
import { CancelButton } from "~/components/utils/CancelButton";
import { useReadOnly } from "~/context/read-only";

interface DatapointActionsProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  variantInferenceIsLoading: boolean;
  onDelete: () => void;
  isDeleting: boolean;
  toggleEditing: () => void;
  onSave: () => void;
  canSave: boolean;
  isEditing: boolean;
  onReset: () => void;
  showTryWithButton: boolean;
  isStale: boolean;
}

export function DatapointActions({
  variants,
  onVariantSelect,
  variantInferenceIsLoading,
  onDelete,
  isDeleting,
  toggleEditing,
  onSave,
  canSave,
  isEditing,
  onReset,
  showTryWithButton,
  isStale,
}: DatapointActionsProps) {
  const { isReadOnly } = useReadOnly();
  const handleCancel = () => {
    onReset();
    toggleEditing();
  };
  return (
    <ActionBar>
      {showTryWithButton && (
        <TryWithButton
          options={variants}
          onOptionSelect={onVariantSelect}
          isLoading={variantInferenceIsLoading}
          disabled={isReadOnly}
        />
      )}
      {isEditing ? (
        <>
          <CancelButton onClick={handleCancel} />
          <SaveButton disabled={!canSave || isReadOnly} onClick={onSave} />
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
