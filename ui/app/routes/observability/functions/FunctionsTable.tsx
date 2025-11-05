import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { FunctionConfig } from "~/types/tensorzero";
import type { FunctionCountInfo } from "~/utils/clickhouse/inference.server";
import { TableItemTime, TableItemFunction } from "~/components/ui/TableItems";
import { toFunctionUrl } from "~/utils/urls";
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

interface MergedFunctionData {
  function_name: string;
  count: number;
  max_timestamp: string;
  type: "chat" | "json" | "?";
  variantsCount: number;
}

const columnHelper = createColumnHelper<MergedFunctionData>();

export default function FunctionsTable({
  functions,
  countsInfo,
}: {
  functions: {
    [x: string]: FunctionConfig | undefined;
  };
  countsInfo: FunctionCountInfo[];
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  // Create a union of all function names from both data sources.
  const mergedFunctions = useMemo(() => {
    const functionNamesSet = new Set<string>([
      ...Object.keys(functions),
      ...countsInfo.map((info) => info.function_name),
    ]);

    return Array.from(functionNamesSet).map((function_name) => {
      const countInfo = countsInfo.find(
        (info) => info.function_name === function_name,
      );
      const function_config = functions[function_name] || null;

      // Special handling: if the function name is 'tensorzero::default', type is 'chat'
      let type: "chat" | "json" | "?";
      if (function_config) {
        type = function_config.type;
      } else {
        type = "?";
      }

      const variantsCount = function_config?.variants
        ? Object.keys(function_config.variants).length
        : 0;

      return {
        function_name,
        count: countInfo ? countInfo.count : 0,
        max_timestamp: countInfo ? countInfo.max_timestamp : "Never",
        type,
        variantsCount,
      };
    });
  }, [functions, countsInfo]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("function_name", {
        header: "Name",
        cell: (info) => (
          <TableItemFunction
            functionName={info.getValue()}
            functionType={info.row.original.type}
            link={toFunctionUrl(info.getValue())}
          />
        ),
      }),
      columnHelper.accessor("variantsCount", {
        header: "Variants",
        cell: (info) => info.getValue(),
      }),
      columnHelper.accessor("count", {
        header: "Inferences",
        cell: (info) => info.getValue(),
      }),
      columnHelper.accessor("max_timestamp", {
        header: "Last Used",
        cell: (info) => {
          const timestamp = info.getValue();
          return timestamp === "Never" ? (
            "Never"
          ) : (
            <TableItemTime timestamp={timestamp} />
          );
        },
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: mergedFunctions,
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
            placeholder="Search functions..."
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
            <TableEmptyState message="No functions found" />
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} id={row.original.function_name}>
                {row.getVisibleCells().map((cell, index) => (
                  <TableCell
                    key={cell.id}
                    className={index === 0 ? "max-w-[200px] lg:max-w-none" : ""}
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
