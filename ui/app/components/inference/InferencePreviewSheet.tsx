import { useEffect, useCallback, useRef } from "react";
import { Link, useFetcher } from "react-router";
import { Maximize2, X } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import {
  InferenceDetailContent,
  type InferenceDetailData,
} from "~/components/inference/InferenceDetailContent";
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
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
      <SheetContent className="w-full overflow-y-auto pb-20 sm:max-w-full md:w-5/6 [&>button.absolute]:hidden">
        <SheetHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <Breadcrumbs
              segments={[
                { label: "Inferences", href: "/observability/inferences" },
              ]}
            />
            <div className="flex items-center gap-4">
              {showFullPageLink && inferenceData && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      to={toInferenceUrl(inferenceData.inference.inference_id)}
                      className="ring-offset-background focus:ring-ring cursor-pointer rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden"
                      aria-label="Open full page"
                    >
                      <Maximize2 className="h-3.5 w-3.5" />
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent>Open full page</TooltipContent>
                </Tooltip>
              )}
              <SheetClose className="ring-offset-background focus:ring-ring cursor-pointer rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden">
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </SheetClose>
            </div>
          </div>
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
