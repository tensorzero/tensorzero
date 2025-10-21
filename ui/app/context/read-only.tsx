"use client";

/**
 * Read-Only Mode Context Provider
 *
 * This module provides a React Context for managing the read-only mode state.
 * When TENSORZERO_UI_READ_ONLY=1 is set in the environment, all write operations
 * (database writes, inference calls) are disabled.
 *
 * @module read-only
 */

import { createContext, use } from "react";

const ReadOnlyContext = createContext(false);
ReadOnlyContext.displayName = "ReadOnlyContext";

/**
 * Hook to check if the application is in read-only mode
 */
export function useReadOnly() {
  return use(ReadOnlyContext);
}

export function ReadOnlyProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: boolean;
}) {
  return <ReadOnlyContext value={value}>{children}</ReadOnlyContext>;
}
