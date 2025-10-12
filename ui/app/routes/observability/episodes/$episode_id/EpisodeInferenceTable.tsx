import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { InferenceByIdRow } from "tensorzero-node";
import { VariantLink } from "~/components/function/variant/VariantLink";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { toFunctionUrl, toInferenceUrl } from "~/utils/urls";

export default function EpisodeInferenceTable({
  inferences,
}: {
  inferences: InferenceByIdRow[];
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
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
              <TableCell className="max-w-[200px]">
                <TableItemShortUuid
                  id={inference.id}
                  link={toInferenceUrl(inference.id)}
                />
              </TableCell>
              <TableCell>
                <TableItemFunction
                  functionName={inference.function_name}
                  functionType={inference.function_type}
                  link={toFunctionUrl(inference.function_name)}
                />
              </TableCell>
              <TableCell>
                <VariantLink
                  variantName={inference.variant_name}
                  functionName={inference.function_name}
                >
                  <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
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
  );
}
