import { useRef } from "react";
import { useEntitySheet } from "~/context/entity-sheet";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";
import { EpisodePreviewSheet } from "~/components/episode/EpisodePreviewSheet";

export function EntitySheet() {
  const { sheetState, closeSheet } = useEntitySheet();

  // Keep the last non-null state so Radix can animate out before unmounting.
  // Without this, setting sheetState to null would immediately unmount the
  // sheet component, skipping the slide-out animation.
  const lastSheetStateRef = useRef(sheetState);
  if (sheetState) {
    lastSheetStateRef.current = sheetState;
  }

  const activeState = lastSheetStateRef.current;
  if (!activeState) return null;

  const isOpen = sheetState !== null;
  const { type } = activeState;

  switch (type) {
    case "inference":
      return (
        <InferencePreviewSheet
          inferenceId={activeState.id}
          isOpen={isOpen}
          onClose={closeSheet}
        />
      );
    case "episode":
      return (
        <EpisodePreviewSheet
          episodeId={activeState.id}
          isOpen={isOpen}
          onClose={closeSheet}
        />
      );
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}
