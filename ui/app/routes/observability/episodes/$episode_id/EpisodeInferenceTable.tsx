import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { VariantLink } from "~/components/function/variant/VariantLink";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { toFunctionUrl, toInferenceUrl } from "~/utils/urls";
import { InferencePreviewSheet } from "~/components/inference/InferencePreviewSheet";
import { Button } from "~/components/ui/button";
import { Eye } from "lucide-react";
import { Suspense } from "react";
import { useLocation, Await, useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import type { InferencesData } from "./route";

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-4 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-20" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-20" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-28" />
          </TableCell>
          <TableCell>
            <Skeleton className="mx-auto h-4 w-8" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

function TableErrorState() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load inferences";

  return (
    <TableRow>
      <TableCell colSpan={5} className="text-center">
        <div className="flex flex-col items-center gap-2 py-8 text-red-600">
          <span className="font-medium">Error loading data</span>
          <span className="text-muted-foreground text-sm">{message}</span>
        </div>
      </TableCell>
    </TableRow>
  );
}

function TableRows({
  data,
  onOpenSheet,
}: {
  data: InferencesData;
  onOpenSheet: (inferenceId: string) => void;
}) {
  const { inferences } = data;

  if (inferences.length === 0) {
    return <TableEmptyState message="No inferences found" />;
  }

  return (
    <>
      {inferences.map((inference) => (
        <TableRow key={inference.inference_id} id={inference.inference_id}>
          <TableCell className="max-w-[200px]">
            <TableItemShortUuid
              id={inference.inference_id}
              link={toInferenceUrl(inference.inference_id)}
            />
          </TableCell>
          <TableCell>
            <TableItemFunction
              functionName={inference.function_name}
              functionType={inference.type}
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
          <TableCell className="text-center">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onOpenSheet(inference.inference_id)}
              aria-label="View inference details"
            >
              <Eye className="h-4 w-4" />
            </Button>
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

interface EpisodeInferenceTableProps {
  data: Promise<InferencesData>;
  onOpenSheet: (inferenceId: string) => void;
  onCloseSheet: () => void;
  openSheetInferenceId: string | null;
}

export default function EpisodeInferenceTable({
  data,
  onOpenSheet,
  onCloseSheet,
  openSheetInferenceId,
}: EpisodeInferenceTableProps) {
  const location = useLocation();

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
            <TableHead className="text-center">Preview</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <Suspense key={location.key} fallback={<SkeletonRows />}>
            <Await resolve={data} errorElement={<TableErrorState />}>
              {(resolvedData) => (
                <TableRows data={resolvedData} onOpenSheet={onOpenSheet} />
              )}
            </Await>
          </Suspense>
        </TableBody>
      </Table>

      <InferencePreviewSheet
        inferenceId={openSheetInferenceId}
        isOpen={!!openSheetInferenceId}
        onClose={onCloseSheet}
      />
    </>
  );
}
