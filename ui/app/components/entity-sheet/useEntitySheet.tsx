import { useCallback } from "react";
import { useSearchParams } from "react-router";

const SHEET_PARAM = "sheet";
const SHEET_ID_PARAM = "sheetId";

type EntitySheetState = { type: "inference"; id: string } | null;

function parseSheetState(searchParams: URLSearchParams): EntitySheetState {
  const type = searchParams.get(SHEET_PARAM);
  const id = searchParams.get(SHEET_ID_PARAM);
  if (type === "inference" && id) {
    return { type, id };
  }
  return null;
}

export function useEntitySheet() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sheetState = parseSheetState(searchParams);

  const openInferenceSheet = useCallback(
    (id: string) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set(SHEET_PARAM, "inference");
          next.set(SHEET_ID_PARAM, id);
          return next;
        },
        { preventScrollReset: true },
      );
    },
    [setSearchParams],
  );

  const closeSheet = useCallback(() => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete(SHEET_PARAM);
        next.delete(SHEET_ID_PARAM);
        return next;
      },
      { preventScrollReset: true },
    );
  }, [setSearchParams]);

  return { sheetState, openInferenceSheet, closeSheet };
}
