import type { GatewayEvent } from "~/types/tensorzero";
import { logger } from "~/utils/logger";

const EVENTS_PER_PAGE = 20;

type FetchOlderEventsResult = {
  items: GatewayEvent[];
  hasMore: boolean;
};

/**
 * Fetches older autopilot events for pagination.
 *
 * Uses the N+1 pagination pattern: requests limit+1 items, if we get more than
 * limit, there are more pages. Returns sorted events (oldest first) with the
 * extra item sliced off.
 *
 * @param sessionId - The autopilot session ID
 * @param beforeEventId - Fetch events before this event ID
 * @param limit - Number of events to return (default: 20)
 * @returns Object with items array and hasMore boolean
 * @throws Error on HTTP errors (enables retry mechanism in useInfiniteScrollUp)
 */
export async function fetchOlderAutopilotEvents(
  sessionId: string,
  beforeEventId: string,
  limit: number = EVENTS_PER_PAGE,
): Promise<FetchOlderEventsResult> {
  const response = await fetch(
    `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events?limit=${limit + 1}&before=${beforeEventId}`,
  );

  if (!response.ok) {
    logger.debug(`API returned ${response.status} when fetching older events`);
    throw new Error(`Failed to fetch older events: ${response.status}`);
  }

  const responseData = (await response.json()) as {
    events: GatewayEvent[];
  };

  logger.debug(
    `Loaded ${responseData.events.length} older events (requested ${limit + 1})`,
  );

  const hasMore = responseData.events.length > limit;

  // Sort by creation time (oldest first) and slice off the extra item if present
  const items = responseData.events
    .sort(
      (a, b) =>
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    )
    .slice(hasMore ? 1 : 0);

  return { items, hasMore };
}
