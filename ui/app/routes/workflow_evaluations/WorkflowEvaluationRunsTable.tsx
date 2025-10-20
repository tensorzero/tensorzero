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
import type { WorkflowEvaluationRunWithEpisodeCount } from "~/utils/clickhouse/workflow_evaluations";
import {
  toWorkflowEvaluationRunUrl,
  toWorkflowEvaluationProjectUrl,
} from "~/utils/urls";

export default function WorkflowEvaluationRunsTable({
  workflowEvaluationRuns,
}: {
  workflowEvaluationRuns: WorkflowEvaluationRunWithEpisodeCount[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>ID</TableHead>
            <TableHead>Project</TableHead>
            <TableHead>Episodes</TableHead>
            <TableHead>Timestamp</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {workflowEvaluationRuns.length === 0 ? (
            <TableEmptyState message="No evaluation runs found" />
          ) : (
            workflowEvaluationRuns.map((run) => (
              <TableRow key={run.id}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={toWorkflowEvaluationRunUrl(run.id)}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {run.name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={toWorkflowEvaluationRunUrl(run.id)}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {run.id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  {run.project_name ? (
                    <Link
                      to={`${toWorkflowEvaluationProjectUrl(run.project_name)}?run_ids=${run.id}`}
                      className="block no-underline"
                    >
                      <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                        {run.project_name}
                      </code>
                    </Link>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </TableCell>
                <TableCell>{run.num_episodes}</TableCell>
                <TableCell>{formatDate(new Date(run.timestamp))}</TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
