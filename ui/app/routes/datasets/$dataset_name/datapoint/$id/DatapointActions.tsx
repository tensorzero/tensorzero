import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
import { EditButton } from "~/components/utils/EditButton";
import { DeleteButton } from "~/components/utils/DeleteButton";

interface DatapointActionsProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  variantInferenceIsLoading: boolean;
  onDelete: () => void;
  isDeleting: boolean;
  showTryWithVariant: boolean;
  className?: string;
}

export function DatapointActions({
  variants,
  onVariantSelect,
  variantInferenceIsLoading,
  onDelete,
  isDeleting,
  showTryWithVariant,
  className,
}: DatapointActionsProps) {
  return (
    <ActionBar className={className}>
      {showTryWithVariant && (
        <TryWithVariantButton
          variants={variants}
          onVariantSelect={onVariantSelect}
          isLoading={variantInferenceIsLoading}
        />
      )}
      <DeleteButton onClick={onDelete} isLoading={isDeleting} />
      <EditButton onClick={() => (window.location.href = "#")} />
    </ActionBar>
  );
}
