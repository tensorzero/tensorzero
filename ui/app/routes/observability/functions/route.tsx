import type { Route } from "./+types/route";
import FunctionsTable from "./FunctionsTable";
import { useConfig } from "~/context/config";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { PageErrorContent } from "~/components/ui/error";
import { Suspense, useMemo, useState } from "react";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { Await, useAsyncError, useLocation } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { StatsBar } from "~/components/ui/StatsBar";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type {
  FunctionConfig,
  FunctionInferenceCount,
} from "~/types/tensorzero";
import { TypeChat, TypeJson } from "~/components/icons/Icons";

export type FunctionsData = {
  countsInfo: FunctionInferenceCount[];
};

function FunctionsPageHeader({ count }: { count?: number }) {
  return <PageHeader heading="Functions" count={count} />;
}

function FunctionsContentSkeleton() {
  return (
    <>
      <FunctionsPageHeader />
      <SectionLayout>
        <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-5 w-44" />
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Variants</TableHead>
              <TableHead>Inferences</TableHead>
              <TableHead>Last Used</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3, 4, 5].map((i) => (
              <TableRow key={i}>
                <TableCell>
                  <Skeleton className="h-4 w-32" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-8" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-16" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-20" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionLayout>
    </>
  );
}

function FunctionsErrorState() {
  const error = useAsyncError();
  return (
    <>
      <FunctionsPageHeader />
      <SectionLayout>
        <PageErrorContent error={error} />
      </SectionLayout>
    </>
  );
}

async function fetchFunctionsData(): Promise<FunctionsData> {
  const httpClient = getTensorZeroClient();
  const countsInfo = await httpClient.listFunctionsWithInferenceCount();
  return { countsInfo };
}

export async function loader() {
  return {
    functionsData: fetchFunctionsData(),
  };
}

function FunctionsSummary({
  functions,
  countsInfo,
}: {
  functions: Record<string, FunctionConfig | undefined>;
  countsInfo: FunctionInferenceCount[];
}) {
  const totalInferences = countsInfo.reduce(
    (acc, info) => acc + info.inference_count,
    0,
  );
  const activeFunctions = countsInfo.filter(
    (info) => info.inference_count > 0,
  ).length;
  const totalFunctions = new Set([
    ...Object.keys(functions),
    ...countsInfo.map((i) => i.function_name),
  ]).size;

  const now = Date.now();
  const oneDayAgo = now - 24 * 60 * 60 * 1000;
  const recentlyActive = countsInfo.filter((info) => {
    if (!info.last_inference_timestamp || info.inference_count === 0)
      return false;
    return new Date(info.last_inference_timestamp).getTime() > oneDayAgo;
  }).length;

  const chatCount = Object.values(functions).filter(
    (f) => f?.type === "chat",
  ).length;
  const jsonCount = Object.values(functions).filter(
    (f) => f?.type === "json",
  ).length;

  return (
    <StatsBar
      items={[
        {
          label: "Active",
          value: `${activeFunctions} / ${totalFunctions}`,
          detail: "with inferences",
        },
        {
          label: "Last 24h",
          value: String(recentlyActive),
          detail: recentlyActive === 1 ? "function used" : "functions used",
        },
        {
          label: "Inferences",
          value: totalInferences.toLocaleString(),
          detail: "total",
        },
        {
          label: "Types",
          custom: (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="bg-bg-type-chat rounded-sm p-0.5">
                  <TypeChat className="text-fg-type-chat" />
                </span>
                <span className="text-fg-primary text-sm font-medium">
                  {chatCount}
                </span>
                <span className="text-fg-muted text-xs">chat</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="bg-bg-type-json rounded-sm p-0.5">
                  <TypeJson className="text-fg-type-json" />
                </span>
                <span className="text-fg-primary text-sm font-medium">
                  {jsonCount}
                </span>
                <span className="text-fg-muted text-xs">json</span>
              </div>
            </div>
          ),
        },
      ]}
    />
  );
}

function FunctionsContent({
  data,
  showInternalFunctions,
  onToggleShowInternalFunctions,
}: {
  data: FunctionsData;
  showInternalFunctions: boolean;
  onToggleShowInternalFunctions: (value: boolean) => void;
}) {
  const { countsInfo } = data;
  const functions = useConfig().functions;
  const [typeFilter, setTypeFilter] = useState<"chat" | "json" | null>(null);

  // Functions filtered only by internal visibility (for summary stats)
  const visibleFunctions = useMemo(() => {
    if (showInternalFunctions) return functions;
    return Object.fromEntries(
      Object.entries(functions).filter(
        ([functionName]) => !functionName.startsWith("tensorzero::"),
      ),
    );
  }, [functions, showInternalFunctions]);

  // Functions filtered by both internal visibility and type (for table)
  const filteredFunctions = useMemo(() => {
    if (!typeFilter) return visibleFunctions;
    return Object.fromEntries(
      Object.entries(visibleFunctions).filter(
        ([, config]) => config?.type === typeFilter,
      ),
    );
  }, [visibleFunctions, typeFilter]);

  const visibleCountsInfo = useMemo(() => {
    if (showInternalFunctions) return countsInfo;
    return countsInfo.filter(
      (info) => !info.function_name.startsWith("tensorzero::"),
    );
  }, [countsInfo, showInternalFunctions]);

  const filteredCountsInfo = useMemo(() => {
    if (!typeFilter) return visibleCountsInfo;
    const filteredNames = new Set(Object.keys(filteredFunctions));
    return visibleCountsInfo.filter((info) =>
      filteredNames.has(info.function_name),
    );
  }, [visibleCountsInfo, typeFilter, filteredFunctions]);

  const displayedFunctionCount = useMemo(() => {
    const functionNames = new Set<string>([
      ...Object.keys(filteredFunctions),
      ...filteredCountsInfo.map((info) => info.function_name),
    ]);

    return functionNames.size;
  }, [filteredCountsInfo, filteredFunctions]);

  return (
    <>
      <FunctionsPageHeader count={displayedFunctionCount} />
      <FunctionsSummary
        functions={visibleFunctions}
        countsInfo={visibleCountsInfo}
      />
      <SectionLayout>
        <FunctionsTable
          functions={filteredFunctions}
          countsInfo={filteredCountsInfo}
          showInternalFunctions={showInternalFunctions}
          onToggleShowInternalFunctions={onToggleShowInternalFunctions}
          typeFilter={typeFilter}
          onTypeFilterChange={setTypeFilter}
        />
      </SectionLayout>
    </>
  );
}

export default function FunctionsPage({ loaderData }: Route.ComponentProps) {
  const { functionsData } = loaderData;
  const location = useLocation();
  const [showInternalFunctions, setShowInternalFunctions] = useState(false);

  return (
    <PageLayout>
      <Suspense key={location.key} fallback={<FunctionsContentSkeleton />}>
        <Await resolve={functionsData} errorElement={<FunctionsErrorState />}>
          {(data) => (
            <FunctionsContent
              data={data}
              showInternalFunctions={showInternalFunctions}
              onToggleShowInternalFunctions={setShowInternalFunctions}
            />
          )}
        </Await>
      </Suspense>
    </PageLayout>
  );
}
