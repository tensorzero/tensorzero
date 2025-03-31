import { Link } from "react-router";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import { FunctionLink } from "~/components/function/FunctionLink";
import { VariantLink } from "~/components/function/variant/VariantLink";
import type { EvalRunInfo } from "~/utils/clickhouse/evaluations";

export default function EvalRunsTable({
  evalRuns,
}: {
  evalRuns: EvalRunInfo[];
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
          {evalRuns.length === 0 ? (
            <TableRow className="hover:bg-bg-primary">
              <TableCell
                colSpan={6}
                className="px-3 py-8 text-center text-fg-muted"
              >
                No evaluation runs found.
              </TableCell>
            </TableRow>
          ) : (
            evalRuns.map((evalRun) => (
              <TableRow key={evalRun.eval_run_id} id={evalRun.eval_run_id}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/evaluations/${evalRun.eval_name}?eval_run_ids=${evalRun.eval_run_id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {evalRun.eval_run_id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/evaluations/${evalRun.eval_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {evalRun.eval_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <Link
                    to={`/datasets/${evalRun.dataset}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {evalRun.dataset}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <FunctionLink functionName={evalRun.function_name}>
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {evalRun.function_name}
                    </code>
                  </FunctionLink>
                </TableCell>
                <TableCell>
                  <VariantLink
                    variantName={evalRun.variant_name}
                    functionName={evalRun.function_name}
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {evalRun.variant_name}
                    </code>
                  </VariantLink>
                </TableCell>
                <TableCell>
                  {formatDate(new Date(evalRun.last_inference_timestamp))}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
