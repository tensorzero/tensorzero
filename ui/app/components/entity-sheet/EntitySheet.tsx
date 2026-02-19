import { useEntitySheet } from "./useEntitySheet";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";

export function EntitySheet() {
  const { sheetState, closeSheet } = useEntitySheet();

  if (!sheetState) return null;

  switch (sheetState.type) {
    case "inference":
      return (
        <InferencePreviewSheet
          inferenceId={sheetState.id}
          isOpen
          onClose={closeSheet}
          showFullPageLink
        />
      );
    default: {
      const _exhaustiveCheck: never = sheetState.type;
      return _exhaustiveCheck;
    }
  }
}
