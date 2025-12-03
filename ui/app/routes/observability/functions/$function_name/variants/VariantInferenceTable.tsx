import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { StoredInference } from "~/types/tensorzero";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import { toInferenceUrl, toEpisodeUrl } from "~/utils/urls";

export default function VariantInferenceTable({
  inferences,
}: {
  inferences: StoredInference[];
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
            <TableRow key={inference.inference_id} id={inference.inference_id}>
              <TableCell className="max-w-[200px]">
                <TableItemShortUuid
                  id={inference.inference_id}
                  link={toInferenceUrl(inference.inference_id)}
                />
              </TableCell>
              <TableCell>
                <TableItemShortUuid
                  id={inference.episode_id}
                  link={toEpisodeUrl(inference.episode_id)}
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
