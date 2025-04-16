import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { useState } from "react";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
} from "~/utils/clickhouse/common";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";
const FF_ENABLE_FEEDBACK =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_FEEDBACK === "1";

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
  inferenceOutput?: ContentBlockOutput[] | JsonInferenceOutput;
  inferenceId: string;
}

export function InferenceActions({
  variants,
  onVariantSelect,
  variantInferenceIsLoading,
  dataset_counts,
  onDatasetSelect,
  hasDemonstration,
  inferenceOutput,
  inferenceId,
}: InferenceActionsProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleModalOpen = () => setIsModalOpen(true);
  const handleModalClose = () => setIsModalOpen(false);

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
      {FF_ENABLE_FEEDBACK && <HumanFeedbackButton onClick={handleModalOpen} />}
      <HumanFeedbackModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        inferenceOutput={inferenceOutput}
        inferenceId={inferenceId}
      />
    </ActionBar>
  );
}
