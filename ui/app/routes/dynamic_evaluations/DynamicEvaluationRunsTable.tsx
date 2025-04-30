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
import { formatDate } from "~/utils/date";
import { FunctionLink } from "~/components/function/FunctionLink";
import { VariantLink } from "~/components/function/variant/VariantLink";
import type { DynamicEvaluationRun } from "~/utils/clickhouse/dynamic_evaluations";

export default function DynamicEvaluationRunsTable({
  evaluationRuns,
}: {
  evaluationRuns: DynamicEvaluationRun[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>ID</TableHead>
            <TableHead>Project</TableHead>
            <TableHead>Timestamp</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {evaluationRuns.length === 0 ? (
            <TableEmptyState message="No evaluation runs found" />
          ) : (
            evaluationRuns.map((evaluationRun) => (
              <TableRow
                key={evaluationRun.evaluation_run_id}
                id={evaluationRun.evaluation_run_id}
              >
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/evaluations/${evaluationRun.evaluation_name}?evaluation_run_ids=${evaluationRun.evaluation_run_id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {evaluationRun.evaluation_run_id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/evaluations/${evaluationRun.evaluation_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {evaluationRun.evaluation_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <Link
                    to={`/datasets/${evaluationRun.dataset_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {evaluationRun.dataset_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <FunctionLink functionName={evaluationRun.function_name}>
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {evaluationRun.function_name}
                    </code>
                  </FunctionLink>
                </TableCell>
                <TableCell>
                  <VariantLink
                    variantName={evaluationRun.variant_name}
                    functionName={evaluationRun.function_name}
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {evaluationRun.variant_name}
                    </code>
                  </VariantLink>
                </TableCell>
                <TableCell>
                  {formatDate(new Date(evaluationRun.last_inference_timestamp))}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
