import { useCallback, useLayoutEffect, useRef, useState } from "react";

/**
 * Smoothly animates an element's height when `trigger` changes.
 * After the CSS transition completes, height resets to "auto" so
 * content that dynamically resizes (e.g. ExpandableElement) isn't clipped.
 */
export function useAnimatedHeight(trigger: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | "auto">("auto");

  // Track the previous trigger value so we only animate on actual changes,
  // not on mount. Using a ref that stores the trigger value (instead of a
  // boolean flag) is StrictMode-safe: StrictMode re-runs effects on mount
  // with the same trigger, so `prevTrigger === trigger` correctly skips.
  const prevTrigger = useRef(trigger);

  // Snapshot of the element's height taken while at "auto", so we have a
  // valid animation start value when the trigger next changes (at which
  // point getBoundingClientRect would already reflect the new content).
  const snapshotHeight = useRef(0);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (prevTrigger.current === trigger) {
      // No trigger change — snapshot the current height when at "auto"
      // so the next animation has a valid starting point.
      if (height === "auto") {
        snapshotHeight.current = el.getBoundingClientRect().height;
      }
      return;
    }
    prevTrigger.current = trigger;

    // When height is a fixed pixel value (mid-animation or pre-transition-end),
    // getBoundingClientRect gives the live rendered height — perfect for rapid
    // step changes. When "auto", the DOM already has the new content, so we
    // fall back to the snapshot from the last stable render.
    const startHeight =
      typeof height === "number"
        ? el.getBoundingClientRect().height
        : snapshotHeight.current;

    // Measure natural height with new content
    el.style.height = "auto";
    const naturalHeight = el.scrollHeight;

    if (startHeight === 0 || startHeight === naturalHeight) {
      snapshotHeight.current = naturalHeight;
      return;
    }

    // Set explicit start height so the CSS transition has a start value
    el.style.height = `${startHeight}px`;
    el.getBoundingClientRect();

    setHeight(naturalHeight);
  }, [trigger, height]);

  const onTransitionEnd = useCallback(() => {
    setHeight("auto");
  }, []);

  return { ref, height, onTransitionEnd };
}
