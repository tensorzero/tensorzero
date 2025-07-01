import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  type VisibilityState,
  type RowSelectionState,
  getSortedRowModel,
  type SortingState,
  type ColumnResizeMode,
} from "@tanstack/react-table";
import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./table";
import { Button } from "./button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "./dropdown-menu";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuCheckboxItem,
} from "./context-menu";
import { 
  ChevronDown, 
  MoreHorizontal, 
  EyeOff, 
  Eye, 
  RotateCcw, 
  ArrowUp, 
  ArrowDown,
  GripVertical
} from "lucide-react";

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  onRowSelect?: (selectedRow: TData | null) => void;
  selectedRowId?: string;
  initialColumnVisibility?: VisibilityState;
}

export function DataTable<TData extends { id?: string }, TValue>({
  columns,
  data,
  onRowSelect,
  selectedRowId,
  initialColumnVisibility = {},
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(initialColumnVisibility);
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [columnSizing, setColumnSizing] = useState({});

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    onColumnSizingChange: setColumnSizing,
    columnResizeMode: "onChange" as ColumnResizeMode,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnSizing,
    },
    enableRowSelection: true,
    enableMultiRowSelection: false, // Single row selection only
    enableColumnResizing: true,
  });

  // Handle row selection changes
  const handleRowClick = (row: { id: string; original: TData }) => {
    if (onRowSelect) {
      const isCurrentlySelected = selectedRowId === row.original.id;
      onRowSelect(isCurrentlySelected ? null : row.original);

      // Update internal selection state
      setRowSelection(isCurrentlySelected ? {} : { [row.id]: true });
    }
  };

  // Handle column context menu actions
  const handleColumnAction = (columnId: string, action: string) => {
    switch (action) {
      case "hide":
        setColumnVisibility((prev) => ({
          ...prev,
          [columnId]: false,
        }));
        break;
      case "show-only":
        // Show only this column and essential columns (first column is usually essential)
        const newVisibility: VisibilityState = {};
        table.getAllColumns().forEach((col) => {
          newVisibility[col.id] = col.id === columnId || col.getIndex() === 0;
        });
        setColumnVisibility(newVisibility);
        break;
      case "reset":
        setColumnVisibility({});
        break;
      case "sort-asc":
        setSorting([{ id: columnId, desc: false }]);
        break;
      case "sort-desc":
        setSorting([{ id: columnId, desc: true }]);
        break;
    }
  };

  return (
    <div className="space-y-4">
      {/* Column visibility controls */}
      {/* <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="ml-auto">
                Columns <ChevronDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {table
                .getAllColumns()
                .filter((column) => column.getCanHide())
                .map((column) => {
                  return (
                    <DropdownMenuCheckboxItem
                      key={column.id}
                      className="capitalize"
                      checked={column.getIsVisible()}
                      onCheckedChange={(value) =>
                        column.toggleVisibility(!!value)
                      }
                    >
                      {column.id}
                    </DropdownMenuCheckboxItem>
                  );
                })}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div> */}

      {/* Table */}
      <div className="rounded-md border">
        <Table style={{ width: table.getCenterTotalSize() }}>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const canSort = header.column.getCanSort();
                  const isSorted = header.column.getIsSorted();

                  return (
                    <ContextMenu key={header.id}>
                      <ContextMenuTrigger asChild>
                        <TableHead 
                          className="group relative cursor-pointer select-none hover:bg-muted/50 transition-colors"
                          style={{ width: header.getSize() }}
                          onClick={(e) => {
                            e.preventDefault();
                            // Trigger context menu on left click
                            const contextEvent = new MouseEvent('contextmenu', {
                              bubbles: true,
                              cancelable: true,
                              clientX: e.clientX,
                              clientY: e.clientY,
                            });
                            e.currentTarget.dispatchEvent(contextEvent);
                          }}
                        >
                          <div className="flex items-center justify-between min-w-0">
                            <div className="truncate mr-3 min-w-0 flex-1">
                              {header.isPlaceholder
                                ? null
                                : flexRender(
                                    header.column.columnDef.header,
                                    header.getContext(),
                                  )}
                            </div>
                            <MoreHorizontal className="h-3 w-3 opacity-0 group-hover:opacity-70 transition-opacity flex-shrink-0" />
                          </div>
                          
                          {/* Resize Handle */}
                          {header.column.getCanResize() && (
                            <div
                              onMouseDown={header.getResizeHandler()}
                              onTouchStart={header.getResizeHandler()}
                              className="absolute top-0 right-0 h-full w-1 cursor-col-resize select-none touch-none bg-transparent hover:bg-blue-500/20 active:bg-blue-500/40 transition-colors group/resize"
                              style={{
                                userSelect: 'none',
                              }}
                            >
                              <div className="absolute top-1/2 right-0 transform -translate-y-1/2 opacity-0 group-hover/resize:opacity-60 transition-opacity">
                                <GripVertical className="h-4 w-4 text-muted-foreground" />
                              </div>
                            </div>
                          )}
                        </TableHead>
                      </ContextMenuTrigger>
                      <ContextMenuContent>
                        <ContextMenuItem
                          onClick={() =>
                            handleColumnAction(header.column.id, "hide")
                          }
                        >
                          <EyeOff className="h-4 w-4" />
                          Hide Column
                        </ContextMenuItem>
                        <ContextMenuItem
                          onClick={() =>
                            handleColumnAction(header.column.id, "show-only")
                          }
                        >
                          <Eye className="h-4 w-4" />
                          Show Only This Column
                        </ContextMenuItem>
                        <ContextMenuSeparator />
                        
                        {/* Column Visibility Submenu */}
                        <ContextMenuSub>
                          <ContextMenuSubTrigger>
                            <Eye className="h-4 w-4" />
                            Show Columns
                          </ContextMenuSubTrigger>
                          <ContextMenuSubContent>
                            {table
                              .getAllColumns()
                              .filter((column) => column.getCanHide())
                              .map((column) => (
                                <ContextMenuCheckboxItem
                                  key={column.id}
                                  className="capitalize"
                                  checked={column.getIsVisible()}
                                  onCheckedChange={(value) =>
                                    column.toggleVisibility(!!value)
                                  }
                                >
                                  {column.id}
                                </ContextMenuCheckboxItem>
                              ))}
                            <ContextMenuSeparator />
                            <ContextMenuItem
                              onClick={() => {
                                table.getAllColumns().forEach((column) => {
                                  if (column.getCanHide()) {
                                    column.toggleVisibility(true);
                                  }
                                });
                              }}
                            >
                              <Eye className="h-4 w-4" />
                              Show All
                            </ContextMenuItem>
                            <ContextMenuItem
                              onClick={() => {
                                table.getAllColumns().forEach((column) => {
                                  if (column.getCanHide() && column.id !== "name") {
                                    column.toggleVisibility(false);
                                  }
                                });
                              }}
                            >
                              <EyeOff className="h-4 w-4" />
                              Hide All
                            </ContextMenuItem>
                          </ContextMenuSubContent>
                        </ContextMenuSub>
                        
                        <ContextMenuSeparator />
                        {canSort && (
                          <>
                            <ContextMenuItem
                              onClick={() =>
                                handleColumnAction(header.column.id, "sort-asc")
                              }
                            >
                              <ArrowUp className="h-4 w-4" />
                              Sort Ascending
                            </ContextMenuItem>
                            <ContextMenuItem
                              onClick={() =>
                                handleColumnAction(
                                  header.column.id,
                                  "sort-desc",
                                )
                              }
                            >
                              <ArrowDown className="h-4 w-4" />
                              Sort Descending
                            </ContextMenuItem>
                            <ContextMenuSeparator />
                          </>
                        )}
                        <ContextMenuItem
                          onClick={() =>
                            handleColumnAction(header.column.id, "reset")
                          }
                        >
                          <RotateCcw className="h-4 w-4" />
                          Reset All Columns
                        </ContextMenuItem>
                      </ContextMenuContent>
                    </ContextMenu>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={
                    selectedRowId === row.original.id ? "selected" : undefined
                  }
                  className={`cursor-pointer ${
                    selectedRowId === row.original.id
                      ? "bg-accent"
                      : "hover:bg-muted/50"
                  }`}
                  onClick={() => handleRowClick(row)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell 
                      key={cell.id}
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
