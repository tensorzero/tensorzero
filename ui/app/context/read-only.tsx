import React, { createContext, useContext } from "react";

type ReadOnlyContextValue = {
  isReadOnly: boolean;
};

const ReadOnlyContext = createContext<ReadOnlyContextValue>({
  isReadOnly: false,
});

/**
 * Provider for read-only mode. Pass a boolean `value` indicating whether the UI
 * should be in read-only mode.
 */
export function ReadOnlyProvider({
  value,
  children,
}: {
  value: boolean;
  children: React.ReactNode;
}) {
  return (
    <ReadOnlyContext.Provider value={{ isReadOnly: value }}>
      {children}
    </ReadOnlyContext.Provider>
  );
}

/**
 * Hook to access read-only mode state.
 */
export function useReadOnly(): ReadOnlyContextValue {
  return useContext(ReadOnlyContext);
