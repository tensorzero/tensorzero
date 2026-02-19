import { useEntitySheet } from "./useEntitySheet";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";

export function EntitySheet() {
  const { sheetState, closeSheet } = useEntitySheet();

  switch (sheetState?.type ?? null) {
    case "inference":
      return (
        <InferencePreviewSheet
          inferenceId={sheetState!.id}
          isOpen
          onClose={closeSheet}
          showFullPageLink
        />
      );
    case null:
      return null;
  }
}
