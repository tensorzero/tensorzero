import { Link } from "react-router";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { VariantLink } from "~/components/function/variant/VariantLink";
import type { EvaluationRunInfo } from "~/types/tensorzero";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { useFunctionConfig } from "~/context/config";
import { toEvaluationRunsUrl, toDatasetUrl, toFunctionUrl } from "~/utils/urls";

function EvaluationRunRow({
  evaluationRun,
}: {
  evaluationRun: EvaluationRunInfo;
}) {
  const functionConfig = useFunctionConfig(evaluationRun.function_name);
  const functionType = functionConfig?.type;

  return (
    <TableRow
      key={evaluationRun.evaluation_run_id}
      id={evaluationRun.evaluation_run_id}
    >
      <TableCell className="max-w-[200px]">
        <TableItemShortUuid
          id={evaluationRun.evaluation_run_id}
          link={toEvaluationRunsUrl(evaluationRun.evaluation_run_id)}
        />
      </TableCell>
      <TableCell className="max-w-[200px]">
        <Link
          to={toEvaluationRunsUrl(evaluationRun.evaluation_run_id)}
          className="block no-underline"
        >
          <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.evaluation_name}
          </code>
        </Link>
      </TableCell>
      <TableCell>
        <Link
          to={toDatasetUrl(evaluationRun.dataset_name)}
          className="block no-underline"
        >
          <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.dataset_name}
          </code>
        </Link>
      </TableCell>
      <TableCell>
        <TableItemFunction
          functionName={evaluationRun.function_name}
          functionType={functionType ?? ""}
          link={toFunctionUrl(
            evaluationRun.function_name,
            evaluationRun.snapshot_hash,
          )}
        />
      </TableCell>
      <TableCell>
        <VariantLink
          variantName={evaluationRun.variant_name}
          functionName={evaluationRun.function_name}
          snapshotHash={evaluationRun.snapshot_hash}
        >
          <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.variant_name}
          </code>
        </VariantLink>
      </TableCell>
      <TableCell>
        <TableItemTime timestamp={evaluationRun.created_at} />
      </TableCell>
    </TableRow>
  );
}

export default function EvaluationRunsTable({
  evaluationRuns,
}: {
  evaluationRuns: EvaluationRunInfo[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run ID</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Dataset</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Created At</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {evaluationRuns.length === 0 ? (
            <TableEmptyState message="No evaluation runs found" />
          ) : (
            evaluationRuns.map((evaluationRun) => (
              <EvaluationRunRow
                key={evaluationRun.evaluation_run_id}
                evaluationRun={evaluationRun}
              />
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
