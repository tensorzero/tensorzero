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
import { Link } from "react-router";
import { FunctionLink } from "~/components/function/FunctionLink";
import { VariantLink } from "~/components/function/variant/VariantLink";
import { TableItemTime } from "~/components/ui/TableItems";

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
  );
}
