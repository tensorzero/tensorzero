import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
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
  showTryWithVariant: boolean;
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
  showTryWithVariant,
}: DatapointActionsProps) {
  const handleCancel = () => {
    onReset();
    toggleEditing();
  };
  return (
    <ActionBar>
      {showTryWithVariant && (
        <TryWithVariantButton
          variants={variants}
          onVariantSelect={onVariantSelect}
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
