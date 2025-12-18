import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { InferenceMetadata } from "~/types/tensorzero";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import { toInferenceUrl, toEpisodeUrl } from "~/utils/urls";
import { uuidv7ToTimestamp } from "~/utils/clickhouse/helpers";

export default function VariantInferenceTable({
  inferences,
}: {
  inferences: InferenceMetadata[];
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
                  link={toInferenceUrl(inference.id)}
                />
              </TableCell>
              <TableCell>
                <TableItemShortUuid
                  id={inference.episode_id}
                  link={toEpisodeUrl(inference.episode_id)}
                />
              </TableCell>
              <TableCell>
                <TableItemTime timestamp={uuidv7ToTimestamp(inference.id)} />
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
