import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
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
                  <TableItemShortUuid
                    id={run.name}
                    link={toWorkflowEvaluationRunUrl(run.id)}
                  />
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid
                    id={run.id}
                    link={toWorkflowEvaluationRunUrl(run.id)}
                  />
                </TableCell>
                <TableCell>
                  {run.project_name ? (
                    <TableItemShortUuid
                      id={run.project_name}
                      link={`${toWorkflowEvaluationProjectUrl(run.project_name)}?run_ids=${run.id}`}
                    />
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </TableCell>
                <TableCell>{run.num_episodes}</TableCell>
                <TableCell>
                  <TableItemTime timestamp={run.timestamp} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}