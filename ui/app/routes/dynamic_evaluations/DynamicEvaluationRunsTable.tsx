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
import type { DynamicEvaluationRunWithEpisodeCount } from "~/utils/clickhouse/dynamic_evaluations";
import { Link } from "~/safe-navigation";

export default function DynamicEvaluationRunsTable({
  dynamicEvaluationRuns,
}: {
  dynamicEvaluationRuns: DynamicEvaluationRunWithEpisodeCount[];
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
          {dynamicEvaluationRuns.length === 0 ? (
            <TableEmptyState message="No evaluation runs found" />
          ) : (
            dynamicEvaluationRuns.map((run) => (
              <TableRow key={run.id}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={[
                      "/dynamic_evaluations/runs/:run_id",
                      { run_id: run.id },
                    ]}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {run.name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={[
                      "/dynamic_evaluations/runs/:run_id",
                      { run_id: run.id },
                    ]}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {run.id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  {run.project_name && (
                    <Link
                      to={{
                        pathname: [
                          "/dynamic_evaluations/projects/:project_name",
                          { project_name: run.project_name },
                        ],
                        search: `?run_ids=${run.id}`,
                      }}
                      className="block no-underline"
                    >
                      <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                        {run.project_name}
                      </code>
                    </Link>
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
