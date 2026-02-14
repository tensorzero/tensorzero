import type { Route } from "./+types/route";
import {
  Await,
  redirect,
  useAsyncError,
  useLocation,
  useNavigate,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import EvaluationRunsTable from "./EvaluationRunsTable";
import { Suspense, useState } from "react";
import { EvaluationsActions } from "./EvaluationsActions";
import LaunchEvaluationModal from "./LaunchEvaluationModal";
import {
  parseEvaluationFormData,
  runEvaluation,
} from "~/utils/evaluations.server";
import { toEvaluationUrl } from "~/utils/urls";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { PageErrorContent } from "~/components/ui/error";
import type { EvaluationRunInfo } from "~/types/tensorzero";

export type EvaluationsData = {
  totalEvaluationRuns: number;
  evaluationRuns: EvaluationRunInfo[];
  offset: number;
  limit: number;
};

function EvaluationsPageHeader({ count }: { count?: number }) {
  return <PageHeader heading="Evaluation Runs" count={count} />;
}

function EvaluationsContentSkeleton() {
  return (
    <>
      <EvaluationsPageHeader />
      <SectionLayout>
        <div className="flex flex-wrap gap-2">
          <Skeleton className="h-8 w-36" />
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Run ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Dataset</TableHead>
              <TableHead>Function</TableHead>
              <TableHead>Variant</TableHead>
              <TableHead>Last Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3, 4, 5].map((i) => (
              <TableRow key={i}>
                <TableCell>
                  <Skeleton className="h-4 w-20" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-32" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-28" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-20" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="mt-4 flex items-center justify-center gap-2">
          <Skeleton className="h-9 w-9 rounded-md" />
          <Skeleton className="h-9 w-9 rounded-md" />
        </div>
      </SectionLayout>
    </>
  );
}

function EvaluationsErrorState() {
  const error = useAsyncError();
  return (
    <>
      <EvaluationsPageHeader />
      <SectionLayout>
        <PageErrorContent error={error} />
      </SectionLayout>
    </>
  );
}

async function fetchEvaluationsData(
  limit: number,
  offset: number,
): Promise<EvaluationsData> {
  const [totalEvaluationRuns, evaluationRunsResponse] = await Promise.all([
    getTensorZeroClient().countEvaluationRuns(),
    getTensorZeroClient().listEvaluationRuns(limit, offset),
  ]);
  return {
    totalEvaluationRuns,
    evaluationRuns: evaluationRunsResponse.runs,
    offset,
    limit,
  };
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "15");

  return {
    evaluationsData: fetchEvaluationsData(limit, offset),
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const evaluationFormData = parseEvaluationFormData({
    evaluation_name: formData.get("evaluation_name"),
    dataset_name: formData.get("dataset_name"),
    variant_name: formData.get("variant_name"),
    concurrency_limit: formData.get("concurrency_limit"),
    inference_cache: formData.get("inference_cache"),
    max_datapoints: formData.get("max_datapoints"),
    precision_targets: formData.get("precision_targets"),
  });

  if (!evaluationFormData) {
    throw new Response("Invalid form data", { status: 400 });
  }

  const {
    evaluation_name,
    dataset_name,
    variant_name,
    concurrency_limit,
    inference_cache,
    max_datapoints,
    precision_targets,
  } = evaluationFormData;

  let evaluation_start_info;
  try {
    evaluation_start_info = await runEvaluation(
      evaluation_name,
      dataset_name,
      variant_name,
      concurrency_limit,
      inference_cache,
      max_datapoints,
      precision_targets,
    );
  } catch (error) {
    logger.error("Error starting evaluation:", error);
    throw new Response(`Failed to start evaluation: ${error}`, {
      status: 500,
    });
  }
  return redirect(
    toEvaluationUrl(evaluation_name, {
      evaluation_run_ids: evaluation_start_info.evaluation_run_id,
    }),
  );
}

function EvaluationsContent({
  data,
  onOpenModal,
}: {
  data: EvaluationsData;
  onOpenModal: () => void;
}) {
  const { totalEvaluationRuns, evaluationRuns, offset, limit } = data;
  const navigate = useNavigate();

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(Math.max(0, offset - limit)));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <EvaluationsPageHeader count={totalEvaluationRuns} />
      <SectionLayout>
        <EvaluationsActions onNewRun={onOpenModal} />
        <EvaluationRunsTable evaluationRuns={evaluationRuns} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + limit >= totalEvaluationRuns}
        />
      </SectionLayout>
    </>
  );
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const { evaluationsData } = loaderData;
  const location = useLocation();
  const [launchEvaluationModalIsOpen, setLaunchEvaluationModalIsOpen] =
    useState(false);

  return (
    <PageLayout>
      <Suspense key={location.key} fallback={<EvaluationsContentSkeleton />}>
        <Await
          resolve={evaluationsData}
          errorElement={<EvaluationsErrorState />}
        >
          {(data) => (
            <EvaluationsContent
              data={data}
              onOpenModal={() => setLaunchEvaluationModalIsOpen(true)}
            />
          )}
        </Await>
      </Suspense>
      <LaunchEvaluationModal
        isOpen={launchEvaluationModalIsOpen}
        onClose={() => setLaunchEvaluationModalIsOpen(false)}
      />
    </PageLayout>
  );
}
