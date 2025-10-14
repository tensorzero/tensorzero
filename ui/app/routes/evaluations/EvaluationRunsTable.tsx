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
import type { EvaluationInfoResult } from "~/utils/clickhouse/evaluations";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { useFunctionConfig } from "~/context/config";
import { toEvaluationUrl, toDatasetUrl, toFunctionUrl } from "~/utils/urls";

function EvaluationRunRow({
  evaluationRun,
}: {
  evaluationRun: EvaluationInfoResult;
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
          link={toEvaluationUrl(evaluationRun.evaluation_name, {
            evaluation_run_ids: evaluationRun.evaluation_run_id,
          })}
        />
      </TableCell>
      <TableCell className="max-w-[200px]">
        <Link
          to={toEvaluationUrl(evaluationRun.evaluation_name)}
          className="block no-underline"
        >
          <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.evaluation_name}
          </code>
        </Link>
      </TableCell>
      <TableCell>
        <Link
          to={toDatasetUrl(evaluationRun.dataset_name)}
          className="block no-underline"
        >
          <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.dataset_name}
          </code>
        </Link>
      </TableCell>
      <TableCell>
        <TableItemFunction
          functionName={evaluationRun.function_name}
          functionType={functionType ?? ""}
          link={toFunctionUrl(evaluationRun.function_name)}
        />
      </TableCell>
      <TableCell>
        <VariantLink
          variantName={evaluationRun.variant_name}
          functionName={evaluationRun.function_name}
        >
          <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
            {evaluationRun.variant_name}
          </code>
        </VariantLink>
      </TableCell>
      <TableCell>
        <TableItemTime timestamp={evaluationRun.last_inference_timestamp} />
      </TableCell>
    </TableRow>
  );
}

export default function EvaluationRunsTable({
  evaluationRuns,
}: {
  evaluationRuns: EvaluationInfoResult[];
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
            <TableHead>Last Updated</TableHead>
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
