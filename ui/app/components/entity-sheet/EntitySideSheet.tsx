import { useEntitySideSheet } from "./EntitySideSheetContext";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";

export function EntitySideSheet() {
  const { sheetState, closeSheet } = useEntitySideSheet();

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
