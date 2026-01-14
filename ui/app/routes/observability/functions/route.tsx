import type { Route } from "./+types/route";
import { isRouteErrorResponse } from "react-router";
import FunctionsTable from "./FunctionsTable";
import { useConfig } from "~/context/config";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { useMemo, useState } from "react";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader() {
  const httpClient = getTensorZeroClient();
  const countsInfo = await httpClient.listFunctionsWithInferenceCount();
  return { countsInfo };
}

export default function FunctionsPage({ loaderData }: Route.ComponentProps) {
  const { countsInfo } = loaderData;
  const functions = useConfig().functions;

  const [showInternalFunctions, setShowInternalFunctions] = useState(false);

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
    <PageLayout>
      <PageHeader
        heading="Functions"
        subheading="LLM application logic with prompts and model configurations"
        count={displayedFunctionCount}
      />
      <SectionLayout>
        <FunctionsTable
          functions={filteredFunctions}
          countsInfo={filteredCountsInfo}
          showInternalFunctions={showInternalFunctions}
          onToggleShowInternalFunctions={setShowInternalFunctions}
        />
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
