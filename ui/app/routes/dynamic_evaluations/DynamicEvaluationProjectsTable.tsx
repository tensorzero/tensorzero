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
import type { DynamicEvaluationProject } from "~/utils/clickhouse/dynamic_evaluations";
import { toDynamicEvaluationProjectUrl } from "~/utils/urls";

export default function DynamicEvaluationProjectsTable({
  dynamicEvaluationProjects,
}: {
  dynamicEvaluationProjects: DynamicEvaluationProject[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Number of Runs</TableHead>
            <TableHead>Last Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {dynamicEvaluationProjects.length === 0 ? (
            <TableEmptyState message="No projects found" />
          ) : (
            dynamicEvaluationProjects.map((project) => (
              <TableRow key={project.name}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={toDynamicEvaluationProjectUrl(project.name)}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                      {project.name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                    {project.count}
                  </code>
                </TableCell>
                <TableCell>
                  {formatDate(new Date(project.last_updated))}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
