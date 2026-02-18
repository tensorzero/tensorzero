import { useInferenceSideSheet } from "./InferenceSideSheetContext";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";

export function InferenceSideSheet() {
  const { inferenceId, closeSheet } = useInferenceSideSheet();

  return (
    <InferencePreviewSheet
      inferenceId={inferenceId}
      isOpen={Boolean(inferenceId)}
      onClose={closeSheet}
      showFullPageLink
    />
  );
}
