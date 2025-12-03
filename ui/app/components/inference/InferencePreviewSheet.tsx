import { useEffect, useCallback, useRef } from "react";
import { useFetcher } from "react-router";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import {
  InferenceDetailContent,
  type InferenceDetailData,
} from "~/components/inference/InferenceDetailContent";
import { toInferenceUrl } from "~/utils/urls";

interface InferencePreviewSheetProps {
  inferenceId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

function getInferenceApiUrl(inferenceId: string) {
  return `/api/inference/${inferenceId}`;
}

export function InferencePreviewSheet({
  inferenceId,
  isOpen,
  onClose,
}: InferencePreviewSheetProps) {
  const fetcher = useFetcher<InferenceDetailData>();

  // Use a ref to access fetcher.load without causing effect re-runs
  // The fetcher object changes identity on state changes, but the load function is stable
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Extract stable values from fetcher for dependency arrays
  const fetcherState = fetcher.state;
  const fetcherDataInferenceId = fetcher.data?.inference.id;

  // Fetch data when sheet opens with an inference ID (only if we don't have data)
  useEffect(() => {
    if (!isOpen || !inferenceId) return;
    // Only fetch if we don't have data or the data is for a different inference
    if (fetcherDataInferenceId === inferenceId) return;
    if (fetcherState !== "idle") return;

    fetcherRef.current.load(getInferenceApiUrl(inferenceId));
  }, [isOpen, inferenceId, fetcherState, fetcherDataInferenceId]);

  const refreshInferenceData = useCallback(() => {
    if (!inferenceId) return;
    fetcherRef.current.load(getInferenceApiUrl(inferenceId));
  }, [inferenceId]);

  const inferenceData = fetcher.data ?? null;
  const isLoading = fetcher.state === "loading";
  const hasError = fetcher.state === "idle" && !fetcher.data;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-full md:w-5/6">
        <SheetHeader>
          <SheetTitle>
            {inferenceData ? (
              <>
                Inference{" "}
                <a
                  href={toInferenceUrl(inferenceData.inference.id)}
                  className="text-md font-mono font-semibold hover:underline"
                >
                  {inferenceData.inference.id}
                </a>
              </>
            ) : (
              "Loading..."
            )}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {isLoading && !inferenceData && (
            <div className="flex items-center justify-center py-12">
              <div className="text-fg-muted text-sm">
                Loading inference details...
              </div>
            </div>
          )}

          {hasError && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">
                Failed to load inference details
              </div>
            </div>
          )}

          {inferenceData && inferenceId && (
            <InferenceDetailContent
              data={inferenceData}
              actionUrl={getInferenceApiUrl(inferenceId)}
              onFeedbackAdded={refreshInferenceData}
            />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
