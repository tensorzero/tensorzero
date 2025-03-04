import { Link } from "react-router";
import type { InferenceByIdRow } from "~/utils/clickhouse/inference";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";

export default function InferencesTable({
  inferences,
}: {
  inferences: InferenceByIdRow[];
}) {
  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Inference ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {inferences.length === 0 ? (
            <TableRow className="hover:bg-background-primary">
              <TableCell
                colSpan={5}
                className="px-3 py-8 text-center text-foreground-muted"
              >
                No inferences found
              </TableCell>
            </TableRow>
          ) : (
            inferences.map((inference) => (
              <TableRow key={inference.id} id={inference.id}>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/observability/inferences/${inference.id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <Link
                    to={`/observability/episodes/${inference.episode_id}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.episode_id}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <Link
                    to={`/observability/functions/${inference.function_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.function_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <Link
                    to={`/observability/functions/${inference.function_name}/variants/${inference.variant_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.variant_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  {formatDate(new Date(inference.timestamp))}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
