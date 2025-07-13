import { createRef, useCallback, useRef } from "react";
import { cn } from "~/utils/common";
import { atom, useAtom } from "jotai";
import { useDrag, type DragConfig } from "../hooks/use-drag";
import { clsx } from "clsx";
import { cva } from "class-variance-authority";

const MIN_HEADER_COL_WIDTH = 200;
const DEFAULT_HEADER_COL_WIDTH = 400;
const MAX_HEADER_COL_WIDTH = 800;

const MIN_HEADER_ROW_HEIGHT = 300;
const DEFAULT_HEADER_ROW_HEIGHT = 500;
const MAX_HEADER_ROW_HEIGHT = 800;

const headerRowHeightAtom = atom(DEFAULT_HEADER_ROW_HEIGHT);
const headerColumnWidthAtom = atom(DEFAULT_HEADER_COL_WIDTH);

const resizableGridElement = createRef<HTMLDivElement>();

/** Horizontal bar that resizes height of header row */
const ResizeHandle: React.FC<
  DragConfig & {
    className?: string;
  }
> = ({ className, ...config }) => {
  const { isDragging, handleMouseDown } = useDrag(config);
  return (
    // TODO Use wrapper so there's a larger touch target!
    <div
      className={clsx(
        "hover:bg-border/50 relative z-30 rounded-full transition",
        isDragging && "bg-blue-500!", // TODO this needs to have higher specificity than `hover:`
        className,
      )}
      onMouseDown={handleMouseDown}
    />
  );
};

export const HorizontalResizeHandle: React.FC<{
  className?: string;
}> = ({ className }) => {
  const [initialRowHeight, setFinalRowHeight] = useAtom(headerRowHeightAtom);
  const rowHeight = useRef<number>(initialRowHeight);
  const resizeRowHeight = useCallback((delta: number) => {
    const value = clamp(
      rowHeight.current + delta,
      MIN_HEADER_ROW_HEIGHT,
      MAX_HEADER_ROW_HEIGHT,
    );
    resizableGridElement.current?.style.setProperty(
      HEADER_ROW_HEIGHT_CSS_VAR,
      `${value}px`,
    );
  }, []);

  return (
    <ResizeHandle
      direction="y"
      className={cn(
        "col-span-full box-content h-1 w-full cursor-row-resize py-0.5",
        resizableGrid({ row: "resize-handle" }),
        className,
      )}
      onDrag={resizeRowHeight}
      onDragEnd={(finalDelta) => {
        rowHeight.current += finalDelta;
        setFinalRowHeight(rowHeight.current);
      }}
    />
  );
};

export const VerticalResizeHandle: React.FC<{
  className?: string;
}> = ({ className }) => {
  const [initialColWidth, setFinalColWidth] = useAtom(headerColumnWidthAtom);
  const colWidth = useRef<number>(initialColWidth);
  const resizeColWidth = useCallback((delta: number) => {
    const value = clamp(
      colWidth.current + delta,
      MIN_HEADER_COL_WIDTH,
      MAX_HEADER_COL_WIDTH,
    );
    resizableGridElement.current?.style.setProperty(
      HEADER_COL_WIDTH_CSS_VAR,
      `${value}px`,
    );
  }, []);

  return (
    <ResizeHandle
      direction="x"
      className={cn(
        "row-span-full box-content h-full w-1 cursor-col-resize px-0.5",
        resizableGrid({ col: "resize-handle" }),
        className,
      )}
      onDrag={resizeColWidth}
      onDragEnd={(finalDelta) => {
        colWidth.current += finalDelta;
        setFinalColWidth(colWidth.current);
      }}
    />
  );
};

export const HEADER_ROW_HEIGHT_CSS_VAR = "--header-row-height";
export const HEADER_COL_WIDTH_CSS_VAR = "--header-col-width";

const GRID_TEMPLATE_COLS = (extraColumnsTemplate: string) => `
[header]        minmax(0, var(${HEADER_COL_WIDTH_CSS_VAR}))
[resize-handle] min-content
[content]       ${extraColumnsTemplate}
`;

const GRID_TEMPLATE_ROWS = `
[header]        var(${HEADER_ROW_HEIGHT_CSS_VAR})
[resize-handle] min-content
[content]       1fr
`;

export const resizableGrid = cva("", {
  variants: {
    row: {
      header: "row-[header]",
      "resize-handle": "row-[resize-handle]",
      content: "row-[content]",
    },
    col: {
      header: "col-[header]",
      "resize-handle": "col-[resize-handle]",
      content: "col-[content]",
    },
  },
});

const clamp = (value: number, min = 0, max = Infinity) =>
  Math.max(min, Math.min(max, value));

const ResizableQuadrant: React.FC<{
  children: React.ReactNode;
  className?: string;
  // /** `grid-template-columns` string to append */
  extraColumnsTemplate: string;
  // /** Number of evenly split content rows (excluding header row) */
  // rowCount: number;
}> = ({ children, className, extraColumnsTemplate }) => (
  <main
    className={cn("grid gap-x-4 gap-y-2", className)}
    style={
      {
        // Initial values
        [HEADER_COL_WIDTH_CSS_VAR]: `${DEFAULT_HEADER_COL_WIDTH}px`,
        [HEADER_ROW_HEIGHT_CSS_VAR]: `${DEFAULT_HEADER_ROW_HEIGHT}px`,

        gridTemplateColumns: GRID_TEMPLATE_COLS(extraColumnsTemplate),
        gridTemplateRows: GRID_TEMPLATE_ROWS,
      } as React.CSSProperties
    }
    ref={resizableGridElement}
  >
    {children}
  </main>
);

export default ResizableQuadrant;
