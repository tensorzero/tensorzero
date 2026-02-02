import { Suspense, useState } from "react";
import { Await, useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import {
  TableErrorNotice,
  getAsyncErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { ModelInferenceItem } from "./ModelInferenceItem";
import { TableItemShortUuid } from "~/components/ui/TableItems";
import type { ModelInferencesData } from "./inference-data.server";

// Section - self-contained with Suspense/Await
interface ModelInferencesSectionProps {
  promise: Promise<ModelInferencesData>;
  locationKey: string;
}

export function ModelInferencesSection({
  promise,
  locationKey,
}: ModelInferencesSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Model Inferences" />
      <Suspense
        key={`model-inferences-${locationKey}`}
        fallback={<ModelInferencesSkeleton />}
      >
        <Await resolve={promise} errorElement={<ModelInferencesError />}>
          {(modelInferences) => (
            <ModelInferencesContent modelInferences={modelInferences} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

// Content (exported for non-streaming InferenceDetailContent)
export function ModelInferencesContent({
  modelInferences,
}: {
  modelInferences: ParsedModelInferenceRow[];
}) {
  const [selectedInference, setSelectedInference] =
    useState<ParsedModelInferenceRow | null>(null);
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const handleRowClick = (inference: ParsedModelInferenceRow) => {
    setSelectedInference(inference);
    setIsSheetOpen(true);
  };

  return (
    <>
      <Table>
        <ModelInferencesTableHeaders />
        <TableBody>
          {modelInferences.length === 0 ? (
            <TableEmptyState message="No model inferences available" />
          ) : (
            modelInferences.map((inference) => (
              <TableRow
                key={inference.id}
                className="hover:bg-bg-hover cursor-pointer"
                onClick={() => handleRowClick(inference)}
              >
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid id={inference.id} />
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <span className="block overflow-hidden font-mono text-ellipsis whitespace-nowrap">
                    {inference.model_name}
                  </span>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
        <SheetContent className="bg-bg-secondary overflow-y-auto p-0">
          {selectedInference && (
            <ModelInferenceItem inference={selectedInference} />
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}

// Shared table headers
function ModelInferencesTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Model</TableHead>
      </TableRow>
    </TableHeader>
  );
}

// Skeleton
function ModelInferencesSkeleton() {
  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        {Array.from({ length: 3 }).map((_, i) => (
          <TableRow key={i}>
            <TableCell>
              <Skeleton className="h-4 w-24" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-20" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// Error
function ModelInferencesError() {
  const error = useAsyncError();
  const message = getAsyncErrorMessage({
    error,
    defaultMessage: "Failed to load model inferences",
  });

  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        <TableRow>
          <TableCell colSpan={2}>
            <TableErrorNotice
              icon={AlertCircle}
              title="Error loading data"
              description={message}
            />
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}
