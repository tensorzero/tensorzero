import { Code } from "~/components/ui/code";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { VariantCounts } from "~/utils/clickhouse/function";
import { VariantLink } from "~/components/function/variant/VariantLink";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  createColumnHelper,
  flexRender,
  type SortingState,
} from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { ChevronUp, ChevronDown, Search } from "lucide-react";

type VariantCountsWithMetadata = VariantCounts & {
  type: string;
};

const columnHelper = createColumnHelper<VariantCountsWithMetadata>();

export default function FunctionVariantTable({
  variant_counts,
  function_name,
}: {
  variant_counts: VariantCountsWithMetadata[];
  function_name: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const columns = useMemo(
    () => [
      columnHelper.accessor("variant_name", {
        header: "Variant Name",
        cell: (info) => (
          <VariantLink
            variantName={info.getValue()}
            functionName={function_name}
          >
            <TableItemShortUuid id={info.getValue()} />
          </VariantLink>
        ),
      }),
      columnHelper.accessor("type", {
        header: "Type",
        cell: (info) => <Code>{info.getValue()}</Code>,
      }),
      columnHelper.accessor("count", {
        header: "Count",
        cell: (info) => info.getValue(),
      }),
      columnHelper.accessor("last_used", {
        header: "Last Used",
        cell: (info) => <TableItemTime timestamp={info.getValue()} />,
      }),
    ],
    [function_name],
  );

  const table = useReactTable({
    data: variant_counts,
    columns,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: "includesString",
  });
  return (
    <div>
      <div className="mb-4">
        <div className="relative">
          <Search className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search variants..."
            value={globalFilter ?? ""}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="border-input bg-background focus:ring-ring w-full rounded-md border py-2 pr-4 pl-10 text-sm focus:border-transparent focus:ring-2 focus:outline-none"
          />
        </div>
      </div>
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className={
                    header.column.getCanSort()
                      ? "cursor-pointer select-none"
                      : ""
                  }
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex items-center gap-1">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                    {header.column.getCanSort() && (
                      <div className="flex flex-col">
                        {header.column.getIsSorted() === "asc" ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <div className="flex flex-col">
                            <ChevronUp className="h-2 w-2 opacity-40" />
                            <ChevronDown className="h-2 w-2 opacity-40" />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length === 0 ? (
            <TableEmptyState message="No variants found" />
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} id={row.original.variant_name}>
                {row.getVisibleCells().map((cell, index) => (
                  <TableCell
                    key={cell.id}
                    className={index === 0 ? "max-w-[200px]" : ""}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
