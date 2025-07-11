import { useState, useCallback, useEffect, useRef } from "react";

export interface DragConfig {
  direction: "x" | "y";
  /** Displacement in pixels along the axis throughout this drag event */
  onDrag: (delta: number) => void;
  onDragEnd: (delta: number) => void;
}

export const useDrag = ({ direction, onDrag, onDragEnd }: DragConfig) => {
  const [isDragging, setIsDragging] = useState(false);
  /** x/y coordinate of initial mouse down */
  const dragStart = useRef(0);
  /** Cumulative pixels dragged from mouse down to mouse up */
  const dragDelta = useRef(0);

  const handleMouseDown = useCallback(
    (event: React.MouseEvent) => {
      // Only drag on left clicks
      if (event.button !== 0) {
        return;
      }

      event.preventDefault();
      setIsDragging(true);
      dragStart.current = direction === "x" ? event.clientX : event.clientY;
      dragDelta.current = 0;
    },
    [direction],
  );

  useEffect(() => {
    if (!isDragging) {
      return;
    }

    const handleMouseMove = (event: MouseEvent) => {
      if (!isDragging) {
        return;
      }

      dragDelta.current =
        (direction === "x" ? event.clientX : event.clientY) - dragStart.current;
      onDrag(dragDelta.current);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      onDragEnd(dragDelta.current);
      dragDelta.current = 0;
      // If `onDragEnd` is called with set dragging set state callback ->
      // "Cannot update a component (`ResizableQuadrant`) while rendering a different component (`ResizeHandle`). To locate the bad setState() call inside `ResizeHandle` ..."
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, onDrag, onDragEnd, direction]);

  return {
    isDragging,
    handleMouseDown,
  };
};
