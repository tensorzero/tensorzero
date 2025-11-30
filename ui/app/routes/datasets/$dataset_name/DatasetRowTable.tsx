import type { Datapoint, DatapointFilter } from "~/types/tensorzero";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
  TableItemText,
} from "~/components/ui/TableItems";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Input } from "~/components/ui/input";
import { Filter, Trash } from "lucide-react";
import { useState, useEffect } from "react";
import { useFetcher, useNavigate } from "react-router";
import { useForm } from "react-hook-form";
import { Form } from "~/components/ui/form";
import { toFunctionUrl, toDatapointUrl, toEpisodeUrl } from "~/utils/urls";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs } from "~/context/config";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import DatapointFilterBuilder from "~/components/querybuilder/DatapointFilterBuilder";

export default function DatasetRowTable({
  rows,
  dataset_name,
  function_name,
  search_query,
  filter,
}: {
  rows: Datapoint[];
  dataset_name: string;
  function_name: string | undefined;
  search_query: string | undefined;
  filter: DatapointFilter | undefined;
}) {
  const activeFetcher = useFetcher();
  const navigate = useNavigate();
  const functions = useAllFunctionConfigs();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterFunctionName, setFilterFunctionName] = useState<string | null>(
    function_name ?? null,
  );
  const [filterSearchQuery, setFilterSearchQuery] = useState(
    search_query ?? "",
  );
  const [datapointToDelete, setDatapointToDelete] = useState<Datapoint | null>(
    null,
  );
  const [filterAdvanced, setFilterAdvanced] = useState<
    DatapointFilter | undefined
  >(filter);

  // Form for the filter sheet (needed for FormLabel in DatapointFilterBuilder)
  const filterForm = useForm();

  // Sync local filter state with props when sheet opens
  useEffect(() => {
    if (filterOpen) {
      setFilterFunctionName(function_name ?? null);
      setFilterSearchQuery(search_query ?? "");
      setFilterAdvanced(filter);
    }
  }, [filterOpen, function_name, search_query, filter]);

  const handleFilterSubmit = () => {
    const searchParams = new URLSearchParams(window.location.search);

    if (filterFunctionName) {
      searchParams.set("function_name", filterFunctionName);
    } else {
      searchParams.delete("function_name");
    }

    if (filterSearchQuery.length > 0) {
      searchParams.set("search_query", filterSearchQuery);
    } else {
      searchParams.delete("search_query");
    }

    if (filterAdvanced) {
      searchParams.set("filter", JSON.stringify(filterAdvanced));
    } else {
      searchParams.delete("filter");
    }

    searchParams.delete("offset");
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
    setFilterOpen(false);
  };

  const handleClearFunctionFilter = () => {
    setFilterFunctionName(null);
  };

  const handleClearSearchFilter = () => {
    setFilterSearchQuery("");
  };

  // Handle successful deletion
  useEffect(() => {
    if (activeFetcher.data && activeFetcher.state === "idle") {
      if (activeFetcher.data.success === true) {
        // Close the dialog
        setDeleteDialogOpen(false);
        setDatapointToDelete(null);
      }
    }
  }, [activeFetcher.data, activeFetcher.state]);

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Updated</TableHead>
            <TableHead className="w-[50px]">
              <div className="flex justify-end">
                <Button
                  variant={
                    function_name || search_query || filter
                      ? "default"
                      : "ghost"
                  }
                  size="iconSm"
                  onClick={() => setFilterOpen(true)}
                >
                  <Filter className="h-4 w-4" />
                </Button>
              </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 ? (
            <TableEmptyState message="No datapoints found" />
          ) : (
            rows.map((row) => (
              <TableRow key={row.id} id={row.id}>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid
                    id={row.id}
                    link={toDatapointUrl(dataset_name, row.id)}
                  />
                </TableCell>
                <TableCell>
                  {row.episode_id && (
                    <TableItemShortUuid
                      id={row.episode_id}
                      link={toEpisodeUrl(row.episode_id)}
                    />
                  )}
                </TableCell>
                <TableCell>
                  {/* TODO: switch to using undefined instead of null */}
                  <TableItemText text={row.name ?? null} />
                </TableCell>
                <TableCell>
                  <TableItemFunction
                    functionName={row.function_name}
                    functionType={row.type}
                    link={toFunctionUrl(row.function_name)}
                  />
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={row.updated_at} />
                </TableCell>
                <TableCell>
                  <div className="text-right">
                    <ReadOnlyGuard
                      asChild
                      onClick={() => {
                        setDatapointToDelete(row);
                        setDeleteDialogOpen(true);
                      }}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-60 transition-opacity hover:opacity-100"
                      >
                        <Trash />
                      </Button>
                    </ReadOnlyGuard>
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Datapoint</DialogTitle>
            <DialogDescription>
              The datapoint will be marked as stale in the database (soft
              deletion). This action cannot be undone.
              {datapointToDelete && (
                <div className="text-muted-foreground mt-2 text-sm">
                  <strong>ID:</strong> {datapointToDelete.id}
                  <br />
                  <strong>Function:</strong> {datapointToDelete.function_name}
                </div>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex justify-between gap-2">
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <div className="flex-1" />
            <Button
              variant="destructive"
              onClick={() => {
                if (datapointToDelete) {
                  const formData = new FormData();
                  formData.append("action", "delete_datapoint");
                  formData.append("datapoint_id", datapointToDelete.id);
                  formData.append(
                    "function_name",
                    datapointToDelete.function_name,
                  );
                  formData.append("function_type", datapointToDelete.type);
                  activeFetcher.submit(formData, {
                    method: "post",
                    action: ".",
                  });
                }
              }}
            >
              <Trash className="inline h-4 w-4" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Sheet open={filterOpen} onOpenChange={setFilterOpen}>
        <SheetContent
          side="right"
          className="flex w-full flex-col md:w-5/6 xl:w-1/2"
        >
          <Form {...filterForm}>
            <SheetHeader>
              <SheetTitle>Filter</SheetTitle>
            </SheetHeader>

            <div className="mt-4 flex-1 space-y-4">
              <div>
                <label className="text-sm font-medium">Function</label>
                <div className="mt-1 flex items-center gap-2">
                  <div className="flex-1">
                    <FunctionSelector
                      selected={filterFunctionName}
                      onSelect={setFilterFunctionName}
                      functions={functions}
                    />
                  </div>
                  {filterFunctionName && (
                    <Button
                      variant="outline"
                      onClick={handleClearFunctionFilter}
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium">Search Query</label>
                  <Badge variant="outline" className="text-xs">
                    Experimental
                  </Badge>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <Input
                    value={filterSearchQuery}
                    onChange={(e) => setFilterSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleFilterSubmit();
                      }
                    }}
                  />
                  <Button variant="outline" onClick={handleClearSearchFilter}>
                    Clear
                  </Button>
                </div>
              </div>

              <div>
                <DatapointFilterBuilder
                  datapointFilter={filterAdvanced}
                  setDatapointFilter={setFilterAdvanced}
                />
              </div>
            </div>

            <SheetFooter className="mt-4">
              <Button onClick={handleFilterSubmit}>Apply Filters</Button>
            </SheetFooter>
          </Form>
        </SheetContent>
      </Sheet>
    </div>
  );
}
