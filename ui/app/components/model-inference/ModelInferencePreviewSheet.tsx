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
import { Breadcrumbs } from "~/components/layout/Breadcrumbs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { toModelInferenceApiUrl, toInferenceUrl } from "~/utils/urls";
import { ModelInferenceItem } from "~/routes/observability/inferences/$inference_id/ModelInferenceItem";
import type { ModelInferenceDetailData } from "~/routes/api/model-inference/$id/route";

interface ModelInferencePreviewSheetProps {
  modelInferenceId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function ModelInferencePreviewSheet({
  modelInferenceId,
  isOpen,
  onClose,
}: ModelInferencePreviewSheetProps) {
  const fetcherKey = modelInferenceId
    ? `model-inference-sheet-${modelInferenceId}`
    : "model-inference-sheet";
  const fetcher = useFetcher<ModelInferenceDetailData>({ key: fetcherKey });

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
    if (!isOpen || !modelInferenceId) return;
    if (fetcherState !== "idle") return;
    if (fetcherData) return;

    hasFetchedRef.current = true;
    fetcherRef.current.load(toModelInferenceApiUrl(modelInferenceId));
  }, [isOpen, modelInferenceId, fetcherState, fetcherData]);

  const hasError =
    fetcherState === "idle" &&
    !fetcherData &&
    modelInferenceId !== null &&
    hasFetchedRef.current;
  const showLoading = !fetcherData && modelInferenceId !== null && !hasError;

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
                  to={toInferenceUrl(fetcherData.inference_id)}
                  className="text-fg-secondary cursor-pointer rounded-sm transition-colors hover:text-orange-600 focus-visible:outline-2 focus-visible:outline-offset-2"
                  aria-label="Open parent inference"
                >
                  <MoveUpRight className="h-4 w-4" />
                </Link>
              </TooltipTrigger>
              <TooltipContent>Open parent inference</TooltipContent>
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
          <Breadcrumbs segments={[{ label: "Model Inferences" }]} />
          <SheetTitle className="font-mono text-2xl font-medium">
            {fetcherData ? (
              <Link
                to={toInferenceUrl(fetcherData.inference_id)}
                className="transition-colors hover:text-orange-600"
              >
                {modelInferenceId}
              </Link>
            ) : modelInferenceId ? (
              modelInferenceId
            ) : (
              "Loading..."
            )}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-8 flex flex-col gap-8">
          {showLoading && (
            <div className="flex items-center justify-center py-12">
              <div className="text-fg-muted text-sm">
                Loading model inference details...
              </div>
            </div>
          )}

          {hasError && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">
                Failed to load model inference details
              </div>
            </div>
          )}

          {fetcherData && (
            <ModelInferenceItem inference={fetcherData.model_inference} />
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
