import { useEffect, useCallback, useRef } from "react";
import { Link, useFetcher } from "react-router";
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
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import { toInferenceUrl } from "~/utils/urls";
import { useToast } from "~/hooks/use-toast";

interface InferencePreviewSheetProps {
  inferenceId: string | null;
  isOpen: boolean;
  onClose: () => void;
  showFullPageLink?: boolean;
}

function getInferenceApiUrl(inferenceId: string) {
  return `/api/inference/${inferenceId}`;
}

export function InferencePreviewSheet({
  inferenceId,
  isOpen,
  onClose,
  showFullPageLink = false,
}: InferencePreviewSheetProps) {
  const fetcher = useFetcher<InferenceDetailData>();
  const { toast } = useToast();

  // Use a ref to access fetcher.load without causing effect re-runs
  // The fetcher object changes identity on state changes, but the load function is stable
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Track the inference ID that was last fetched to detect when it changes
  const lastFetchedInferenceIdRef = useRef<string | null>(null);

  // Extract stable values from fetcher for dependency arrays
  const fetcherState = fetcher.state;
  const fetcherDataInferenceId = fetcher.data?.inference.inference_id;

  // Fetch data when sheet opens with an inference ID (only if we don't have data)
  // Also refetch when inference ID changes to avoid showing stale data
  useEffect(() => {
    if (!isOpen || !inferenceId) return;

    // Check if inference ID changed - if so, always refetch
    const inferenceIdChanged =
      lastFetchedInferenceIdRef.current !== inferenceId;
    lastFetchedInferenceIdRef.current = inferenceId;

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
  // Only show error if we're idle, have no data, but have an inferenceId we should have fetched
  // This avoids showing error on initial render before the fetch starts
  const hasError =
    fetcher.state === "idle" &&
    !fetcher.data &&
    inferenceId !== null &&
    lastFetchedInferenceIdRef.current === inferenceId;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full overflow-y-auto pb-20 sm:max-w-full md:w-5/6">
        {showFullPageLink && inferenceData && (
          <Link
            to={toInferenceUrl(inferenceData.inference.inference_id)}
            className="text-fg-secondary absolute top-4 right-14 text-sm transition-colors hover:text-orange-600"
          >
            Open full page
          </Link>
        )}
        <SheetHeader className="gap-3">
          <Breadcrumbs
            segments={[
              { label: "Inferences", href: "/observability/inferences" },
            ]}
          />
          <SheetTitle className="font-mono text-2xl font-medium">
            {inferenceId ?? "Loading..."}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-8 flex flex-col gap-8">
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
