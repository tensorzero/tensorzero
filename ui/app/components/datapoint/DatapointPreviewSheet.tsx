import { useEffect, useRef } from "react";
import { Link, useFetcher } from "react-router";
import { MoveUpRight, X } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import { DatapointDetailContent } from "~/components/datapoint/DatapointDetailContent";
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { toDatapointApiUrl, toDatapointUrl } from "~/utils/urls";
import type { DatapointDetailData } from "~/routes/api/datapoint/$id/route";

interface DatapointPreviewSheetProps {
  datapointId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function DatapointPreviewSheet({
  datapointId,
  isOpen,
  onClose,
}: DatapointPreviewSheetProps) {
  const fetcher = useFetcher<DatapointDetailData>();

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const lastFetchedDatapointIdRef = useRef<string | null>(null);

  const fetcherState = fetcher.state;
  const fetcherDataDatapointId = fetcher.data?.datapoint.id;

  useEffect(() => {
    if (!isOpen || !datapointId) return;

    const datapointIdChanged =
      lastFetchedDatapointIdRef.current !== datapointId;
    lastFetchedDatapointIdRef.current = datapointId;

    if (
      !datapointIdChanged &&
      fetcherDataDatapointId === datapointId &&
      fetcherState === "idle"
    ) {
      return;
    }

    if (fetcherState !== "idle") return;

    fetcherRef.current.load(toDatapointApiUrl(datapointId));
  }, [isOpen, datapointId, fetcherState, fetcherDataDatapointId]);

  const datapointData = fetcher.data ?? null;
  const isLoading = fetcher.state === "loading";
  const hasError =
    fetcher.state === "idle" &&
    !fetcher.data &&
    datapointId !== null &&
    lastFetchedDatapointIdRef.current === datapointId;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        aria-describedby={undefined}
        onOpenAutoFocus={(e) => e.preventDefault()}
        className="pt-page-top pb-page-bottom w-full overflow-y-auto border-l-0 px-8 focus:outline-hidden sm:max-w-full md:w-5/6 [&>button.absolute]:hidden"
      >
        <div className="absolute top-8 right-8 z-10 flex items-center gap-5">
          {datapointData && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to={toDatapointUrl(
                    datapointData.dataset_name,
                    datapointData.datapoint.id,
                  )}
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
          <Breadcrumbs segments={[{ label: "Datapoints" }]} />
          <SheetTitle className="font-mono text-2xl font-medium">
            {datapointId ? (
              datapointData ? (
                <Link
                  to={toDatapointUrl(
                    datapointData.dataset_name,
                    datapointData.datapoint.id,
                  )}
                  className="transition-colors hover:text-orange-600"
                >
                  {datapointId}
                </Link>
              ) : (
                datapointId
              )
            ) : (
              "Loading..."
            )}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-8 flex flex-col gap-8">
          {isLoading && !datapointData && (
            <div className="flex items-center justify-center py-12">
              <div className="text-fg-muted text-sm">
                Loading datapoint details...
              </div>
            </div>
          )}

          {hasError && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">
                Failed to load datapoint details
              </div>
            </div>
          )}

          {datapointData && datapointId && (
            <DatapointDetailContent data={datapointData} />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
