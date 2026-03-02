import { createContext, useContext, createElement } from "react";

/**
 * Provides a config snapshot hash to descendant components.
 *
 * Pages that view historical data (inferences, evaluations) set this so
 * outbound links to config-defined entities (functions, variants) include
 * `?snapshot_hash=X`. The target page then loads the historical config
 * instead of the current one.
 *
 * DB entities (inferences, episodes) carry their own snapshot_hash, so
 * links to those pages do NOT need this — their loaders read the hash
 * from the entity itself.
 */
const SnapshotHashContext = createContext<string | null>(null);

export function SnapshotHashProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: string | null;
}) {
  return createElement(SnapshotHashContext.Provider, { value }, children);
}

export function useSnapshotHash(): string | null {
  return useContext(SnapshotHashContext);
}
