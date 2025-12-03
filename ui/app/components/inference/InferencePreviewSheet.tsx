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
import { useToast } from "~/hooks/use-toast";

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
  const { toast } = useToast();

  // Use a ref to access fetcher.load without causing effect re-runs
  // The fetcher object changes identity on state changes, but the load function is stable
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Track the previous inference ID to detect when it changes
  const prevInferenceIdRef = useRef<string | null>(null);

  // Extract stable values from fetcher for dependency arrays
  const fetcherState = fetcher.state;
  const fetcherDataInferenceId = fetcher.data?.inference.id;

  // Fetch data when sheet opens with an inference ID (only if we don't have data)
  // Also refetch when inference ID changes to avoid showing stale data
  useEffect(() => {
    if (!isOpen || !inferenceId) return;

    // Check if inference ID changed - if so, always refetch
    const inferenceIdChanged = prevInferenceIdRef.current !== inferenceId;
    prevInferenceIdRef.current = inferenceId;

    // Only fetch if we don't have data, data is for a different inference, or ID changed
    if (
      !inferenceIdChanged &&
      fetcherDataInferenceId === inferenceId &&
      fetcherState === "idle"
    ) {
      return;
    }

    if (fetcherState !== "idle") return;

    fetcherRef.current.load(getInferenceApiUrl(inferenceId));
  }, [isOpen, inferenceId, fetcherState, fetcherDataInferenceId]);

  const refreshInferenceData = useCallback(
    (redirectUrl?: string) => {
      if (!inferenceId) return;
      // Show success toast when feedback is added
      toast.success({ title: "Feedback Added" });
      // Load with newFeedbackId if provided in the redirect URL for proper polling
      if (redirectUrl) {
        fetcherRef.current.load(redirectUrl);
      } else {
        fetcherRef.current.load(getInferenceApiUrl(inferenceId));
      }
    },
    [inferenceId, toast],
  );

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
              onFeedbackAdded={refreshInferenceData}
            />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
