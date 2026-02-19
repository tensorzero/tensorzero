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
  const fetcher = useFetcher<ModelInferenceDetailData>();

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const lastFetchedIdRef = useRef<string | null>(null);

  const fetcherState = fetcher.state;
  const fetcherDataId = fetcher.data?.model_inference.id;

  useEffect(() => {
    if (!isOpen || !modelInferenceId) return;

    const idChanged = lastFetchedIdRef.current !== modelInferenceId;
    lastFetchedIdRef.current = modelInferenceId;

    if (
      !idChanged &&
      fetcherDataId === modelInferenceId &&
      fetcherState === "idle"
    ) {
      return;
    }

    if (fetcherState !== "idle") return;

    fetcherRef.current.load(toModelInferenceApiUrl(modelInferenceId));
  }, [isOpen, modelInferenceId, fetcherState, fetcherDataId]);

  const data = fetcher.data ?? null;
  const isLoading = fetcher.state === "loading";
  const hasError =
    fetcher.state === "idle" &&
    !fetcher.data &&
    modelInferenceId !== null &&
    lastFetchedIdRef.current === modelInferenceId;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        aria-describedby={undefined}
        onOpenAutoFocus={(e) => e.preventDefault()}
        className="pt-page-top pb-page-bottom w-full overflow-y-auto border-l-0 px-8 focus:outline-hidden sm:max-w-full md:w-5/6 [&>button.absolute]:hidden"
      >
        <div className="absolute top-8 right-8 z-10 flex items-center gap-5">
          {data && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to={toInferenceUrl(data.inference_id)}
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
            {data ? (
              <Link
                to={toInferenceUrl(data.inference_id)}
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
          {isLoading && !data && (
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

          {data && <ModelInferenceItem inference={data.model_inference} />}
        </div>
      </SheetContent>
    </Sheet>
  );
}
