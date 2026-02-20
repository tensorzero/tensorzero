import { useEffect, useCallback, useRef } from "react";
import { Link, useFetcher } from "react-router";
import { MoveUpRight, X } from "lucide-react";
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
import { toInferenceApiUrl, toInferenceUrl } from "~/utils/urls";
import { useToast } from "~/hooks/use-toast";

interface InferencePreviewSheetProps {
  inferenceId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function InferencePreviewSheet({
  inferenceId,
  isOpen,
  onClose,
}: InferencePreviewSheetProps) {
  const fetcherKey = inferenceId
    ? `inference-sheet-${inferenceId}`
    : "inference-sheet";
  const fetcher = useFetcher<InferenceDetailData>({ key: fetcherKey });
  const { toast } = useToast();

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Track whether we've initiated a fetch for the current key to distinguish
  // "haven't fetched yet" (show loading) from "fetched and failed" (show error)
  const hasFetchedRef = useRef(false);
  const prevKeyRef = useRef(fetcherKey);
  if (prevKeyRef.current !== fetcherKey) {
    prevKeyRef.current = fetcherKey;
    hasFetchedRef.current = false;
  }

  const fetcherState = fetcher.state;
  const fetcherData = fetcher.data;

  useEffect(() => {
    if (!isOpen || !inferenceId) return;
    if (fetcherState !== "idle") return;
    if (fetcherData) return;

    hasFetchedRef.current = true;
    fetcherRef.current.load(toInferenceApiUrl(inferenceId));
  }, [isOpen, inferenceId, fetcherState, fetcherData]);

  const refreshInferenceData = useCallback(
    (redirectUrl?: string) => {
      if (!inferenceId) return;
      toast.success({ title: "Feedback Added" });
      if (redirectUrl) {
        fetcherRef.current.load(redirectUrl);
      } else {
        fetcherRef.current.load(toInferenceApiUrl(inferenceId));
      }
    },
    [inferenceId, toast],
  );

  const hasError =
    fetcherState === "idle" &&
    !fetcherData &&
    inferenceId !== null &&
    hasFetchedRef.current;
  const showLoading = !fetcherData && inferenceId !== null && !hasError;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        aria-describedby={undefined}
        onOpenAutoFocus={(e) => e.preventDefault()}
        className="pt-page-top pb-page-bottom w-full overflow-y-auto border-l-0 px-8 focus:outline-hidden sm:max-w-full md:w-5/6 [&>button.absolute]:hidden"
      >
        <div className="absolute top-8 right-8 z-10 flex items-center gap-5">
          {fetcherData && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to={toInferenceUrl(fetcherData.inference.inference_id)}
                  className="text-fg-secondary cursor-pointer rounded-sm transition-colors hover:text-orange-600 focus-visible:outline-2 focus-visible:outline-offset-2"
                  aria-label="Open full page"
                >
                  <MoveUpRight className="h-4 w-4" />
                </Link>
              </TooltipTrigger>
              <TooltipContent>Open full page</TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <SheetClose className="text-fg-secondary cursor-pointer rounded-sm transition-colors hover:text-orange-600 focus-visible:outline-2 focus-visible:outline-offset-2">
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </SheetClose>
            </TooltipTrigger>
            <TooltipContent>Close</TooltipContent>
          </Tooltip>
        </div>

        <SheetHeader className="space-y-3">
          <Breadcrumbs
            segments={[
              { label: "Inferences", href: "/observability/inferences" },
            ]}
          />
          <SheetTitle className="font-mono text-2xl font-medium">
            {inferenceId ? (
              <Link
                to={toInferenceUrl(inferenceId)}
                className="transition-colors hover:text-orange-600"
              >
                {inferenceId}
              </Link>
            ) : (
              "Loading..."
            )}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-8 flex flex-col gap-8">
          {showLoading && (
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

          {fetcherData && inferenceId && (
            <InferenceDetailContent
              data={fetcherData}
              onFeedbackAdded={refreshInferenceData}
            />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
