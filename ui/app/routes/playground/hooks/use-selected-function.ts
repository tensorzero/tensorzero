import { useAtom } from "jotai";
import { useCallback, useEffect } from "react";
import { useSearchParams } from "react-router";
import { lastViewedFunctionAtom } from "../state";

export const useSelectedFunction = () => {
  /**
   * Prefer browser history for current function state - store function in query param.
   *
   * However, if user opens the Playground directly/in a new tab,
   * default to the mostly recently viewed function.
   */
  const [lastViewedFunction, setLastViewedFunction] = useAtom(
    lastViewedFunctionAtom,
  );

  const [searchParams, setSearchParams] = useSearchParams();
  const setSelectedFunction = useCallback(
    (functionName: string) => setSearchParams({ function: functionName }),
    [setSearchParams],
  );

  const functionName = searchParams.get("function") ?? undefined;
  useEffect(() => {
    // Navigate to ?function={lastViewedFunction} if nothing is in the URL
    if (!functionName && lastViewedFunction) {
      setSelectedFunction(lastViewedFunction);
    }
  }, [functionName, lastViewedFunction, setSelectedFunction]);
  useEffect(() => {
    // On function selection, synchronize back to localStorage
    if (functionName) {
      setLastViewedFunction(functionName);
    }
  }, [functionName, setLastViewedFunction]);

  return [functionName, setSelectedFunction] as const;
};
