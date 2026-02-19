import { useEffect, useRef } from "react";
import { useFetcher } from "react-router";
import { X } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import { FeedbackDetailContent } from "~/components/feedback/FeedbackDetailContent";
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { FeedbackDetailData } from "~/routes/api/feedback-detail/$id/route";

interface FeedbackPreviewSheetProps {
  feedbackId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

function getFeedbackApiUrl(feedbackId: string) {
  return `/api/feedback-detail/${feedbackId}`;
}

export function FeedbackPreviewSheet({
  feedbackId,
  isOpen,
  onClose,
}: FeedbackPreviewSheetProps) {
  const fetcher = useFetcher<FeedbackDetailData>();

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const lastFetchedFeedbackIdRef = useRef<string | null>(null);

  const fetcherState = fetcher.state;
  const fetcherDataId = fetcher.data?.id;

  useEffect(() => {
    if (!isOpen || !feedbackId) return;

    const feedbackIdChanged = lastFetchedFeedbackIdRef.current !== feedbackId;
    lastFetchedFeedbackIdRef.current = feedbackId;

    if (
      !feedbackIdChanged &&
      fetcherDataId === feedbackId &&
      fetcherState === "idle"
    ) {
      return;
    }

    if (fetcherState !== "idle") return;

    fetcherRef.current.load(getFeedbackApiUrl(feedbackId));
  }, [isOpen, feedbackId, fetcherState, fetcherDataId]);

  const feedbackData = fetcher.data ?? null;
  const isLoading = fetcher.state === "loading";
  const hasError =
    fetcher.state === "idle" &&
    !fetcher.data &&
    feedbackId !== null &&
    lastFetchedFeedbackIdRef.current === feedbackId;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        aria-describedby={undefined}
        onOpenAutoFocus={(e) => e.preventDefault()}
        className="pt-page-top pb-page-bottom w-full overflow-y-auto border-l-0 px-8 focus:outline-hidden sm:max-w-full md:w-5/6 [&>button.absolute]:hidden"
      >
        <div className="absolute top-8 right-8 z-10 flex items-center gap-5">
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
          <Breadcrumbs segments={[{ label: "Feedback" }]} />
          <SheetTitle className="font-mono text-2xl font-medium">
            {feedbackId ?? "Loading..."}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-8 flex flex-col gap-8">
          {isLoading && !feedbackData && (
            <div className="flex items-center justify-center py-12">
              <div className="text-fg-muted text-sm">
                Loading feedback details...
              </div>
            </div>
          )}

          {hasError && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">
                Failed to load feedback details
              </div>
            </div>
          )}

          {feedbackData && feedbackId && (
            <FeedbackDetailContent data={feedbackData} />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
