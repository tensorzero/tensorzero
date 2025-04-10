import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";

interface InferenceActionsProps {
  variants: string[];
  onVariantSelect: (variant: string) => void;
  variantInferenceIsLoading: boolean;
  dataset_counts: DatasetCountInfo[];
  onDatasetSelect: (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
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
}: InferenceActionsProps) {
  return (
    <ActionBar>
      <TryWithVariantButton
        variants={variants}
        onVariantSelect={onVariantSelect}
        isLoading={variantInferenceIsLoading}
      />
      {FF_ENABLE_DATASETS && (
        <AddToDatasetButton
          dataset_counts={dataset_counts}
          onDatasetSelect={onDatasetSelect}
          hasDemonstration={hasDemonstration}
        />
      )}
    </ActionBar>
  );
}
