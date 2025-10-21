/**
 * Read-Only Mode Context Provider
 *
 * This module provides a React Context for managing the read-only mode state.
 * When TENSORZERO_UI_READ_ONLY=1 is set in the environment, all write operations
 * (database writes, inference calls) are disabled.
 *
 * @module read-only
 */

import { createContext, useContext } from "react";

interface ReadOnlyContextValue {
  isReadOnly: boolean;
}

const ReadOnlyContext = createContext<ReadOnlyContextValue | null>(null);
ReadOnlyContext.displayName = "ReadOnlyContext";

/**
 * Hook to check if the application is in read-only mode
 * @returns Object with isReadOnly boolean
 */
export function useReadOnly(): ReadOnlyContextValue {
  const context = useContext(ReadOnlyContext);
  if (!context) {
    throw new Error("useReadOnly must be used within a ReadOnlyProvider");
  }
  return context;
}

export function ReadOnlyProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: boolean;
}) {
  return (
    <ReadOnlyContext.Provider value={{ isReadOnly: value }}>
      {children}
    </ReadOnlyContext.Provider>
  );
}
