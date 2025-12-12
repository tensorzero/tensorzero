/**
 * Result type for pagination with cursor-based navigation.
 */
export type PaginationResult<T> = {
  items: T[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

/**
 * Applies cursor-based pagination logic to a result set.
 *
 * This implements the "limit + 1" pattern where you fetch one extra item
 * to detect if there are more pages. The function handles the logic for
 * determining page availability and slicing the results correctly based
 * on pagination direction.
 *
 * Pagination direction semantics (assuming results are ordered newest-first):
 * - `before`: Going to older items (next page). Extra item is at the end.
 * - `after`: Going to newer items (previous page). Extra item is at position 0.
 * - Neither: Initial page load showing most recent items.
 *
 * @param results - The raw results from the query (should have limit + 1 items if more exist)
 * @param limit - The requested page size
 * @param options - Cursor options indicating pagination direction
 * @returns Paginated items with hasNextPage/hasPreviousPage flags
 */
export function applyPaginationLogic<T>(
  results: T[],
  limit: number,
  options: { before?: string | null; after?: string | null },
): PaginationResult<T> {
  const hasMore = results.length > limit;
  const { before, after } = options;

  if (before) {
    // Going backwards in time (older). hasMore means there are older pages.
    // We came from a newer page, so there's always a previous (newer) page.
    // Extra item is at the end, so take first 'limit' items.
    return {
      items: results.slice(0, limit),
      hasNextPage: hasMore,
      hasPreviousPage: true,
    };
  } else if (after) {
    // Going forwards in time (newer). hasMore means there are newer pages.
    // We came from an older page, so there's always a next (older) page.
    // Extra item is at position 0, so take items from position 1 onwards.
    return {
      items: hasMore ? results.slice(1, limit + 1) : results,
      hasNextPage: true,
      hasPreviousPage: hasMore,
    };
  } else {
    // Initial page load - showing most recent.
    // Extra item is at the end, so take first 'limit' items.
    return {
      items: results.slice(0, limit),
      hasNextPage: hasMore,
      hasPreviousPage: false,
    };
  }
}
