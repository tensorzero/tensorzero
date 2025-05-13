import type { Route } from "./+types/route";
import { isRouteErrorResponse } from "react-router";
import FunctionsTable from "./FunctionsTable";
import { useConfig } from "~/context/config";
import { countInferencesByFunction } from "~/utils/clickhouse/inference.server";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";

export async function loader() {
  const countsInfo = await countInferencesByFunction();
  return { countsInfo };
}

export default function FunctionsPage({ loaderData }: Route.ComponentProps) {
  const { countsInfo } = loaderData;
  const functions = useConfig().functions;
  const totalFunctions = Object.keys(functions).length;

  return (
    <PageLayout>
      <PageHeader heading="Functions" count={totalFunctions} />
      <SectionLayout>
        <FunctionsTable functions={functions} countsInfo={countsInfo} />
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

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
