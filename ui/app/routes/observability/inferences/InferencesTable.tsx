import type { InferenceFilter, StoredInference } from "~/types/tensorzero";
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
  TableItemShortUuid,
  TableItemTime,
  TableItemFunction,
} from "~/components/ui/TableItems";
import { VariantLink } from "~/components/function/variant/VariantLink";
import { toInferenceUrl, toEpisodeUrl, toFunctionUrl } from "~/utils/urls";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Input } from "~/components/ui/input";
import { Filter } from "lucide-react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { useForm } from "react-hook-form";
import { Form } from "~/components/ui/form";
import {
  Sheet,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs } from "~/context/config";
import InferenceFilterBuilder from "~/components/querybuilder/InferenceFilterBuilder";

export default function InferencesTable({
  inferences,
  function_name,
  variant_name,
  episode_id,
  search_query,
  filter,
}: {
  inferences: StoredInference[];
  function_name: string | undefined;
  variant_name: string | undefined;
  episode_id: string | undefined;
  search_query: string | undefined;
  filter: InferenceFilter | undefined;
}) {
  const navigate = useNavigate();
  const functions = useAllFunctionConfigs();

  const [filterOpen, setFilterOpen] = useState(false);

  // Local state for filter form
  const [filterFunctionName, setFilterFunctionName] = useState<string | null>(
    function_name ?? null,
  );
  const [filterVariantName, setFilterVariantName] = useState(
    variant_name ?? "",
  );
  const [filterEpisodeId, setFilterEpisodeId] = useState(episode_id ?? "");
  const [filterSearchQuery, setFilterSearchQuery] = useState(
    search_query ?? "",
  );
  const [filterAdvanced, setFilterAdvanced] = useState<
    InferenceFilter | undefined
  >(filter);

  // Form for the filter sheet (needed for FormLabel in InferenceFilterBuilder)
  const filterForm = useForm();

  // Sync local filter state with props when sheet opens
  useEffect(() => {
    if (filterOpen) {
      setFilterFunctionName(function_name ?? null);
      setFilterVariantName(variant_name ?? "");
      setFilterEpisodeId(episode_id ?? "");
      setFilterSearchQuery(search_query ?? "");
      setFilterAdvanced(filter);
    }
  }, [
    filterOpen,
    function_name,
    variant_name,
    episode_id,
    search_query,
    filter,
  ]);

  const handleFilterSubmit = () => {
    const searchParams = new URLSearchParams();

    if (filterFunctionName) {
      searchParams.set("function_name", filterFunctionName);
    }

    if (filterVariantName.length > 0) {
      searchParams.set("variant_name", filterVariantName);
    }

    if (filterEpisodeId.length > 0) {
      searchParams.set("episode_id", filterEpisodeId);
    }

    if (filterSearchQuery.length > 0) {
      searchParams.set("search_query", filterSearchQuery);
    }

    if (filterAdvanced) {
      searchParams.set("filter", JSON.stringify(filterAdvanced));
    }

    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
    setFilterOpen(false);
  };

  const handleClearFunctionFilter = () => {
    setFilterFunctionName(null);
  };

  const handleClearVariantFilter = () => {
    setFilterVariantName("");
  };

  const handleClearEpisodeFilter = () => {
    setFilterEpisodeId("");
  };

  const handleClearSearchFilter = () => {
    setFilterSearchQuery("");
  };

  const hasActiveFilters =
    function_name || variant_name || episode_id || search_query || filter;

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Inference ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
            <TableHead className="w-[50px]">
              <div className="flex justify-end">
                <Button
                  variant={hasActiveFilters ? "default" : "ghost"}
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
          {inferences.length === 0 ? (
            <TableEmptyState message="No inferences found" />
          ) : (
            inferences.map((inference) => (
              <TableRow
                key={inference.inference_id}
                id={inference.inference_id}
              >
                <TableCell>
                  <TableItemShortUuid
                    id={inference.inference_id}
                    link={toInferenceUrl(inference.inference_id)}
                  />
                </TableCell>
                <TableCell>
                  <TableItemShortUuid
                    id={inference.episode_id}
                    link={toEpisodeUrl(inference.episode_id)}
                  />
                </TableCell>
                <TableCell>
                  <TableItemFunction
                    functionName={inference.function_name}
                    functionType={inference.type}
                    link={toFunctionUrl(inference.function_name)}
                  />
                </TableCell>
                <TableCell>
                  <VariantLink
                    variantName={inference.variant_name}
                    functionName={inference.function_name}
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {inference.variant_name}
                    </code>
                  </VariantLink>
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={inference.timestamp} />
                </TableCell>
                <TableCell />
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Sheet open={filterOpen} onOpenChange={setFilterOpen}>
        <SheetContent
          side="right"
          className="flex w-full flex-col md:w-5/6 xl:w-1/2"
        >
          <Form {...filterForm}>
            <SheetHeader>
              <SheetTitle>Filter</SheetTitle>
            </SheetHeader>

            <div className="mt-4 flex min-h-0 flex-1 flex-col space-y-4 overflow-y-auto">
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
                <label className="text-sm font-medium">Variant</label>
                <div className="mt-1 flex items-center gap-2">
                  <Input
                    value={filterVariantName}
                    onChange={(e) => setFilterVariantName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleFilterSubmit();
                      }
                    }}
                    placeholder="Enter variant name"
                  />
                  {filterVariantName && (
                    <Button
                      variant="outline"
                      onClick={handleClearVariantFilter}
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">Episode ID</label>
                <div className="mt-1 flex items-center gap-2">
                  <Input
                    value={filterEpisodeId}
                    onChange={(e) => setFilterEpisodeId(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleFilterSubmit();
                      }
                    }}
                    placeholder="Enter episode ID"
                  />
                  {filterEpisodeId && (
                    <Button
                      variant="outline"
                      onClick={handleClearEpisodeFilter}
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
                    placeholder="Search in input and output"
                  />
                  {filterSearchQuery && (
                    <Button variant="outline" onClick={handleClearSearchFilter}>
                      Clear
                    </Button>
                  )}
                </div>
              </div>

              <div>
                <InferenceFilterBuilder
                  inferenceFilter={filterAdvanced}
                  setInferenceFilter={setFilterAdvanced}
                />
              </div>
            </div>

            <SheetFooter className="mt-4 shrink-0">
              <Button onClick={handleFilterSubmit}>Apply Filters</Button>
            </SheetFooter>
          </Form>
        </SheetContent>
      </Sheet>
    </div>
  );
}
