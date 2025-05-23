import type { InferenceByIdRow } from "~/utils/clickhouse/inference";
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
import { FunctionLink } from "~/components/function/FunctionLink";
import { VariantLink } from "~/components/function/variant/VariantLink";

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
            <TableEmptyState message="No inferences found" />
          ) : (
            inferences.map((inference) => (
              <TableRow key={inference.id} id={inference.id}>
                <TableCell>
                  <TableItemShortUuid
                    id={inference.id}
                    link={`/observability/inferences/${inference.id}`}
                  />
                </TableCell>
                <TableCell>
                  <TableItemShortUuid
                    id={inference.episode_id}
                    link={`/observability/episodes/${inference.episode_id}`}
                  />
                </TableCell>
                <TableCell>
                  <FunctionLink functionName={inference.function_name}>
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.function_name}
                    </code>
                  </FunctionLink>
                </TableCell>
                <TableCell>
                  <VariantLink
                    variantName={inference.variant_name}
                    functionName={inference.function_name}
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {inference.variant_name}
                    </code>
                  </VariantLink>
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={inference.timestamp} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
