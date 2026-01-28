import { useEffect, useRef, useState } from "react";

/**
 * Hook to measure an element's height dynamically using ResizeObserver.
 * Returns a ref to attach to the element and the current height.
 */
export function useElementHeight(initialHeight: number = 0) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [height, setHeight] = useState(initialHeight);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const measureHeight = () => {
      const newHeight = element.offsetHeight;
      if (newHeight > 0) {
        setHeight((prev) => (prev !== newHeight ? newHeight : prev));
      }
    };

    measureHeight();

    const resizeObserver = new ResizeObserver(measureHeight);
    resizeObserver.observe(element);

    return () => resizeObserver.disconnect();
  }, []);

  return [ref, height] as const;
}
