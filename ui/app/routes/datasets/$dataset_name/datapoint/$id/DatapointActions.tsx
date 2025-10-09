import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithButton } from "~/components/inference/TryWithButton";
import { EditButton } from "~/components/utils/EditButton";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { SaveButton } from "~/components/utils/SaveButton";
import { CancelButton } from "~/components/utils/CancelButton";

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
}: DatapointActionsProps) {
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
        />
      )}
      {isEditing ? (
        <>
          <CancelButton onClick={handleCancel} />
          <SaveButton disabled={!canSave} onClick={onSave} />
        </>
      ) : (
        <EditButton onClick={toggleEditing} />
      )}
      <DeleteButton onClick={onDelete} isLoading={isDeleting} />
    </ActionBar>
  );
}
