import { describe, it, expect } from "vitest";

/**
 * Tests for EventStream rendering logic
 *
 * The EventStream component has conditional rendering for:
 * 1. "Session started" divider - shows when we've reached the start AND there's content
 * 2. Loading skeletons - shows when we haven't reached the start yet
 */

// Pure function extracted from EventStream rendering logic
function computeEventStreamState({
  hasReachedStart,
  isLoadingOlder,
  loadError,
  eventsCount,
  optimisticMessagesCount,
}: {
  hasReachedStart: boolean;
  isLoadingOlder: boolean;
  loadError: boolean;
  eventsCount: number;
  optimisticMessagesCount: number;
}) {
  const hasContent = eventsCount > 0 || optimisticMessagesCount > 0;
  const showSessionStart =
    (hasReachedStart || optimisticMessagesCount > 0) &&
    !isLoadingOlder &&
    !loadError &&
    hasContent;
  const showSkeletons = !showSessionStart && !hasReachedStart && !loadError;
  const showSentinel = !showSessionStart;

  return {
    showSessionStart,
    showSkeletons,
    showSentinel,
  };
}

describe("EventStream rendering logic", () => {
  describe("showSessionStart", () => {
    it("should not show session start on empty session even when hasReachedStart is true", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(false);
    });

    it("should show session start when hasReachedStart is true and has events", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(true);
    });

    it("should show session start when has optimistic messages (even without events)", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 1,
      });

      expect(result.showSessionStart).toBe(true);
    });

    it("should not show session start while loading older messages", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: true,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(false);
    });

    it("should not show session start when there is a load error", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: true,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(false);
    });
  });

  describe("showSkeletons", () => {
    it("should show skeletons when not reached start yet", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSkeletons).toBe(true);
    });

    it("should not show skeletons when hasReachedStart is true (empty session)", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 0,
      });

      expect(result.showSkeletons).toBe(false);
    });

    it("should not show skeletons when hasReachedStart is true (with events)", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSkeletons).toBe(false);
    });

    it("should not show skeletons when there is a load error", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: true,
        eventsCount: 0,
        optimisticMessagesCount: 0,
      });

      expect(result.showSkeletons).toBe(false);
    });

    it("should not show skeletons when showing session start", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      // When showSessionStart is true, showSkeletons should be false
      expect(result.showSessionStart).toBe(true);
      expect(result.showSkeletons).toBe(false);
    });
  });

  describe("showSentinel", () => {
    it("should show sentinel when session start is not shown", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSentinel).toBe(true);
    });

    it("should not show sentinel when session start is shown", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 5,
        optimisticMessagesCount: 0,
      });

      expect(result.showSentinel).toBe(false);
    });
  });

  describe("combined scenarios", () => {
    it("empty session: no divider, no skeletons, show sentinel", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(false);
      expect(result.showSkeletons).toBe(false);
      expect(result.showSentinel).toBe(true);
    });

    it("loading initial: no divider, show skeletons, show sentinel", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(false);
      expect(result.showSkeletons).toBe(true);
      expect(result.showSentinel).toBe(true);
    });

    it("session with content: show divider, no skeletons, no sentinel", () => {
      const result = computeEventStreamState({
        hasReachedStart: true,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 10,
        optimisticMessagesCount: 0,
      });

      expect(result.showSessionStart).toBe(true);
      expect(result.showSkeletons).toBe(false);
      expect(result.showSentinel).toBe(false);
    });

    it("user typing (optimistic message only): show divider", () => {
      const result = computeEventStreamState({
        hasReachedStart: false,
        isLoadingOlder: false,
        loadError: false,
        eventsCount: 0,
        optimisticMessagesCount: 1,
      });

      expect(result.showSessionStart).toBe(true);
    });
  });
});
