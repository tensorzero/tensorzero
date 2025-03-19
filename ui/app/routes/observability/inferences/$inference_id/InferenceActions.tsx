import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";

interface InferenceActionsProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  variantInferenceIsLoading: boolean;
  dataset_counts: DatasetCountInfo[];
  onDatasetSelect: (
    dataset: string,
    output: "inference" | "demonstration" | "none",
  ) => void;
  hasDemonstration: boolean;
  className?: string;
}

export function InferenceActions({
  variants,
  onVariantSelect,
  variantInferenceIsLoading,
  dataset_counts,
  onDatasetSelect,
  hasDemonstration,
  className,
}: InferenceActionsProps) {
  return (
    <ActionBar className={className}>
      <TryWithVariantButton
        variants={variants}
        onVariantSelect={onVariantSelect}
        isLoading={variantInferenceIsLoading}
      />
      <AddToDatasetButton
        dataset_counts={dataset_counts}
        onDatasetSelect={onDatasetSelect}
        hasDemonstration={hasDemonstration}
      />
    </ActionBar>
  );
}
