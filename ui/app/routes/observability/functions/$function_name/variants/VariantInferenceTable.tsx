import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { InferenceByIdRow } from "~/utils/clickhouse/inference";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";

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
        {inferences.length === 0 ? (
          <TableEmptyState message="No inferences found" />
        ) : (
          inferences.map((inference) => (
            <TableRow key={inference.id} id={inference.id}>
              <TableCell className="max-w-[200px]">
                <TableItemShortUuid
                  id={inference.id}
                  link={[
                    "/observability/inferences/:inference_id",
                    { inference_id: inference.id },
                  ]}
                />
              </TableCell>
              <TableCell>
                <TableItemShortUuid
                  id={inference.episode_id}
                  link={[
                    "/observability/episodes/:episode_id",
                    { episode_id: inference.episode_id },
                  ]}
                />
              </TableCell>
              <TableCell>
                <TableItemTime timestamp={inference.timestamp} />
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
