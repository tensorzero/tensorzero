import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { InferenceByIdRow, ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { VariantLink } from "~/components/function/variant/VariantLink";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { toFunctionUrl, toInferenceUrl } from "~/utils/urls";
import { InferenceDetailSheet } from "~/components/inference/InferenceDetailSheet";
import { Button } from "~/components/ui/button";
import { Eye } from "lucide-react";

export default function EpisodeInferenceTable({
  inferences,
  onInferenceHover,
  onOpenSheet,
  onCloseSheet,
  getInferenceData,
  isInferenceLoading,
  getError,
  openSheetInferenceId,
}: {
  inferences: InferenceByIdRow[];
  onInferenceHover?: (inferenceId: string) => void;
  onOpenSheet?: (inferenceId: string) => void;
  onCloseSheet?: () => void;
  getInferenceData?: (inferenceId: string) => ParsedInferenceRow | null;
  isInferenceLoading?: (inferenceId: string) => boolean;
  getError?: (inferenceId: string) => string | null;
  openSheetInferenceId?: string | null;
}) {
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
            <TableHead>View</TableHead>
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
                <TableCell>
                  {onOpenSheet && onInferenceHover && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onOpenSheet(inference.id)}
                      onMouseEnter={() => onInferenceHover(inference.id)}
                      aria-label="View inference details"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      {openSheetInferenceId && (
        <InferenceDetailSheet
          inference={getInferenceData?.(openSheetInferenceId) ?? null}
          isLoading={isInferenceLoading?.(openSheetInferenceId) ?? false}
          error={getError?.(openSheetInferenceId) ?? null}
          isOpen={!!openSheetInferenceId}
          onClose={onCloseSheet!}
        />
      )}
    </>
  );
}
