import { useLayoutEffect, useRef, useState } from "react";

/**
 * Smoothly animates an element's height when `trigger` changes.
 * Returns a ref to attach to the container and the current height value.
 */
export function useAnimatedHeight(trigger: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | undefined>(undefined);
  const isFirstRender = useRef(true);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Temporarily remove explicit height so scrollHeight reflects content
    const prevHeight = el.style.height;
    el.style.height = "auto";
    const naturalHeight = el.scrollHeight;
    el.style.height = prevHeight;

    if (isFirstRender.current) {
      isFirstRender.current = false;
      setHeight(naturalHeight);
      return;
    }
    // Force reflow at the old height so the transition has a start value
    el.getBoundingClientRect();
    setHeight(naturalHeight);
  }, [trigger]);

  return { ref, height };
}
