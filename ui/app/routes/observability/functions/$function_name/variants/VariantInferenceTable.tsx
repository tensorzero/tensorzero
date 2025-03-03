import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { InferenceByIdRow } from "~/utils/clickhouse/inference";
import { formatDate } from "~/utils/date";
import { Link } from "react-router";

export default function VariantInferenceTable({
  inferences,
}: {
  inferences: InferenceByIdRow[];
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Episode ID</TableHead>
          <TableHead>Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {inferences.map((inference) => (
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
            <TableCell>
              <Link
                to={`/observability/episodes/${inference.episode_id}`}
                className="block no-underline"
              >
                <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                  {inference.episode_id}
                </code>
              </Link>
            </TableCell>
            <TableCell>{formatDate(new Date(inference.timestamp))}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
