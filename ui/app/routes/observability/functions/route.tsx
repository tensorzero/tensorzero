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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { FunctionInferenceCount } from "~/types/tensorzero";

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

  const filteredFunctions = useMemo(() => {
    if (showInternalFunctions) return functions;

    return Object.fromEntries(
      Object.entries(functions).filter(
        ([functionName]) => !functionName.startsWith("tensorzero::"),
      ),
    );
  }, [functions, showInternalFunctions]);

  const filteredCountsInfo = useMemo(() => {
    if (showInternalFunctions) return countsInfo;

    return countsInfo.filter(
      (info) => !info.function_name.startsWith("tensorzero::"),
    );
  }, [countsInfo, showInternalFunctions]);

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
      <SectionLayout>
        <FunctionsTable
          functions={filteredFunctions}
          countsInfo={filteredCountsInfo}
          showInternalFunctions={showInternalFunctions}
          onToggleShowInternalFunctions={onToggleShowInternalFunctions}
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
