import type { Route } from "./+types/route";
import FunctionsTable from "./FunctionsTable";
import { useConfig } from "~/context/config";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
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
      <PageHeader heading="Functions" count={displayedFunctionCount} />
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
  return <LayoutErrorBoundary error={error} />;
}
