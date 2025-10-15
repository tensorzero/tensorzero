import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { Button } from "~/components/ui/button";
import { Eye } from "lucide-react";
import type { InferenceByIdRow, ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { VariantLink } from "~/components/function/variant/VariantLink";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { toFunctionUrl, toInferenceUrl } from "~/utils/urls";
import { InferencePeek } from "~/components/inference/InferencePeek";

export default function EpisodeInferenceTable({
  inferences,
  onInferencePeek,
}: {
  inferences: InferenceByIdRow[];
  onInferencePeek?: (inferenceId: string) => void;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Function</TableHead>
          <TableHead>Variant</TableHead>
          <TableHead>Time</TableHead>
          {onInferencePeek && <TableHead className="w-20">Actions</TableHead>}
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
              {onInferencePeek && (
                <TableCell>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onInferencePeek(inference.id);
                    }}
                    className="h-8 w-8 p-0"
                  >
                    <Eye className="h-4 w-4" />
                    <span className="sr-only">Peek at inference details</span>
                  </Button>
                </TableCell>
              )}
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
