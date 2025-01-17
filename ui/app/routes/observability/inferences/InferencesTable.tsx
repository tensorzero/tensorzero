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
          {inferences.map((inference) => (
            <TableRow key={inference.id} id={inference.id}>
              <TableCell className="max-w-[200px]">
                <a
                  href={`/observability/inference/${inference.id}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.id}
                  </code>
                </a>
              </TableCell>
              <TableCell className="max-w-[200px]">
                <a
                  href={`/observability/episode/${inference.episode_id}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.episode_id}
                  </code>
                </a>
              </TableCell>
              <TableCell>
                <a
                  href={`/observability/function/${inference.function_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.function_name}
                  </code>
                </a>
              </TableCell>
              <TableCell>
                <a
                  href={`#${inference.variant_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.variant_name}
                  </code>
                </a>
              </TableCell>
              <TableCell>{formatDate(new Date(inference.timestamp))}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
