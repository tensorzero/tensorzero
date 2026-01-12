import type { Route } from "./+types/route";
import FunctionsTable from "./FunctionsTable";
import { useAllFunctionConfigs } from "~/context/config";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { useMemo, useState } from "react";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

export async function loader() {
  try {
    const httpClient = getTensorZeroClient();
    const countsInfo = await httpClient.listFunctionsWithInferenceCount();
    return { countsInfo };
  } catch (error) {
    logger.error("Failed to load functions", error);
    return { countsInfo: [] };
  }
}

const EMPTY_FUNCTIONS = {};

export default function FunctionsPage({ loaderData }: Route.ComponentProps) {
  const { countsInfo } = loaderData;
  const functionsConfig = useAllFunctionConfigs();
  const functions = functionsConfig ?? EMPTY_FUNCTIONS;

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
