import { useCallback, useLayoutEffect, useRef, useState } from "react";

/**
 * Smoothly animates an element's height when `trigger` changes.
 * After the CSS transition completes, height resets to "auto" so
 * content that dynamically resizes (e.g. ExpandableElement) isn't clipped.
 */
export function useAnimatedHeight(trigger: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | "auto">("auto");
  const isFirstRender = useRef(true);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    // Capture actual rendered height (works whether style is "auto" or pixels)
    const currentHeight = el.getBoundingClientRect().height;

    // Measure natural height with new content
    el.style.height = "auto";
    const naturalHeight = el.scrollHeight;

    // Set explicit start height so the CSS transition has a start value
    el.style.height = `${currentHeight}px`;
    el.getBoundingClientRect();

    setHeight(naturalHeight);
  }, [trigger]);

  const onTransitionEnd = useCallback(() => {
    setHeight("auto");
  }, []);

  return { ref, height, onTransitionEnd };
}
