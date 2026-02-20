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
  EpisodeDetailContent,
  type EpisodeDetailData,
} from "~/components/episode/EpisodeDetailContent";
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { toEpisodeApiUrl, toEpisodeUrl } from "~/utils/urls";
import { useToast } from "~/hooks/use-toast";

interface EpisodePreviewSheetProps {
  episodeId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function EpisodePreviewSheet({
  episodeId,
  isOpen,
  onClose,
}: EpisodePreviewSheetProps) {
  const fetcherKey = episodeId ? `episode-sheet-${episodeId}` : "episode-sheet";
  const fetcher = useFetcher<EpisodeDetailData>({ key: fetcherKey });
  const { toast } = useToast();

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const hasFetchedRef = useRef(false);
  const prevKeyRef = useRef(fetcherKey);
  if (prevKeyRef.current !== fetcherKey) {
    prevKeyRef.current = fetcherKey;
    hasFetchedRef.current = false;
  }

  const fetcherState = fetcher.state;
  const fetcherData = fetcher.data;

  useEffect(() => {
    if (!isOpen || !episodeId) return;
    if (fetcherState !== "idle") return;
    if (fetcherData) return;

    hasFetchedRef.current = true;
    fetcherRef.current.load(toEpisodeApiUrl(episodeId));
  }, [isOpen, episodeId, fetcherState, fetcherData]);

  const refreshEpisodeData = useCallback(
    (redirectUrl?: string) => {
      if (!episodeId) return;
      toast.success({ title: "Feedback Added" });
      // The feedback action returns a page URL for episodes, but we need the API
      // URL for fetcher.load(). Extract newFeedbackId for ClickHouse polling.
      let apiUrl = toEpisodeApiUrl(episodeId);
      if (redirectUrl) {
        const newFeedbackId = new URL(
          redirectUrl,
          window.location.origin,
        ).searchParams.get("newFeedbackId");
        if (newFeedbackId) {
          apiUrl += `?newFeedbackId=${newFeedbackId}`;
        }
      }
      fetcherRef.current.load(apiUrl);
    },
    [episodeId, toast],
  );

  const hasError =
    fetcherState === "idle" &&
    !fetcherData &&
    episodeId !== null &&
    hasFetchedRef.current;
  const showLoading = !fetcherData && episodeId !== null && !hasError;

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
                  to={toEpisodeUrl(fetcherData.episode_id)}
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
            segments={[{ label: "Episodes", href: "/observability/episodes" }]}
          />
          <SheetTitle className="font-mono text-2xl font-medium">
            {episodeId ? (
              <Link
                to={toEpisodeUrl(episodeId)}
                className="transition-colors hover:text-orange-600"
              >
                {episodeId}
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
                Loading episode details...
              </div>
            </div>
          )}

          {hasError && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">
                Failed to load episode details
              </div>
            </div>
          )}

          {fetcherData && episodeId && (
            <EpisodeDetailContent
              data={fetcherData}
              onFeedbackAdded={refreshEpisodeData}
            />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
