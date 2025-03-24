import {
  queryInferenceTable,
  queryInferenceTableBounds,
  countInferencesByFunction,
} from "~/utils/clickhouse/inference";
import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { countTotalEvalRuns } from "~/utils/clickhouse/evaluations.server";
import InferenceSearchBar from "../observability/inferences/InferenceSearchBar";

export async function loader({ request }: Route.LoaderArgs) {
  const totalEvalRuns = await countTotalEvalRuns();
  return {
    totalEvalRuns,
  };
}

export default function EvalSummaryPage({ loaderData }: Route.ComponentProps) {
  const navigate = useNavigate();
  const { totalEvalRuns } = loaderData;

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];

  const handleNextPage = () => {
    navigate(`?before=${bottomInference.id}&pageSize=${pageSize}`, {
      preventScrollReset: true,
    });
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topInference.id}&pageSize=${pageSize}`, {
      preventScrollReset: true,
    });
  };

  const disablePrevious =
    !bounds?.last_id || bounds.last_id === topInference.id;
  const disableNext =
    !bounds?.first_id || bounds.first_id === bottomInference.id;

  return (
    <PageLayout>
      <PageHeader heading="Evaluation Runs" count={totalEvalRuns} />
      <SectionLayout>
        <InferenceSearchBar />
        <InferencesTable inferences={inferences} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={disablePrevious}
          disableNext={disableNext}
        />
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
