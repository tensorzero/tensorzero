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
        <TableItemShortUuid
          id={evaluationRun.evaluation_name}
          link={toEvaluationUrl(evaluationRun.evaluation_name)}
        />
      </TableCell>
      <TableCell>
        <TableItemShortUuid
          id={evaluationRun.dataset_name}
          link={toDatasetUrl(evaluationRun.dataset_name)}
        />
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
          <TableItemShortUuid id={evaluationRun.variant_name} />
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
