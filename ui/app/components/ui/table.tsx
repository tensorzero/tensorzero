import * as React from "react";
import { useAsyncError } from "react-router";

import { cn } from "~/utils/common";

const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <div className="border-border relative w-full overflow-auto rounded-md border">
    <table
      ref={ref}
      className={cn("bg-bg-primary w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
));
Table.displayName = "Table";

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("[&_tr]:border-b", className)} {...props} />
));
TableHeader.displayName = "TableHeader";

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0", className)}
    {...props}
  />
));
TableBody.displayName = "TableBody";

const TableFooter = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tfoot
    ref={ref}
    className={cn(
      "bg-bg-secondary text-fg-secondary border-t font-medium last:[&>tr]:border-b-0",
      className,
    )}
    {...props}
  />
));
TableFooter.displayName = "TableFooter";

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "data-[state=selected]:bg-muted h-12 border-b transition-colors",
      className,
    )}
    {...props}
  />
));
TableRow.displayName = "TableRow";

const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "bg-bg-secondary text-fg-tertiary h-10 px-3 text-left align-middle font-medium [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
      className,
    )}
    {...props}
  />
));
TableHead.displayName = "TableHead";

const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn(
      "px-3 py-2.5 align-middle [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
      className,
    )}
    {...props}
  />
));
TableCell.displayName = "TableCell";

const TableCaption = React.forwardRef<
  HTMLTableCaptionElement,
  React.HTMLAttributes<HTMLTableCaptionElement>
>(({ className, ...props }, ref) => (
  <caption
    ref={ref}
    className={cn("text-muted-foreground mt-4 text-sm", className)}
    {...props}
  />
));
TableCaption.displayName = "TableCaption";

interface TableEmptyStateProps
  extends React.TdHTMLAttributes<HTMLTableCellElement> {
  message?: string;
}

const TableEmptyState = React.forwardRef<
  HTMLTableCellElement,
  TableEmptyStateProps
>(({ message = "No data found", ...props }, ref) => (
  <TableRow>
    <TableCell
      ref={ref}
      colSpan={1000}
      className="text-fg-muted py-10 text-center"
      {...props}
    >
      {message}
    </TableCell>
  </TableRow>
));
TableEmptyState.displayName = "TableEmptyState";

interface TableAsyncErrorStateProps {
  colSpan: number;
  defaultMessage?: string;
}

/**
 * Error state for tables using React Router's <Await> component.
 * Must be rendered inside an <Await errorElement={...}> to access the async error.
 * @throws Error if used outside of an <Await errorElement={...}> context
 */
function TableAsyncErrorState({
  colSpan,
  defaultMessage = "Failed to load data",
}: TableAsyncErrorStateProps) {
  const error = useAsyncError();

  if (error === undefined) {
    throw new Error(
      "TableAsyncErrorState must be used inside an <Await errorElement={...}>",
    );
  }

  const message = error instanceof Error ? error.message : defaultMessage;

  return (
    <TableRow>
      <TableCell colSpan={colSpan} className="text-center">
        <div className="flex flex-col items-center gap-2 py-8 text-red-600">
          <span className="font-medium">Error loading data</span>
          <span className="text-muted-foreground text-sm">{message}</span>
        </div>
      </TableCell>
    </TableRow>
  );
}

export {
  Table,
  TableAsyncErrorState,
  TableBody,
  TableCaption,
  TableCell,
  TableEmptyState,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
};
