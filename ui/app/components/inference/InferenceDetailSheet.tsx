import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import Input from "~/components/inference/Input";
import { Output } from "~/components/inference/Output";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { formatDateWithSeconds } from "~/utils/date";

interface InferenceDetailSheetProps {
  inference: ParsedInferenceRow | null;
  isLoading?: boolean;
  error?: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function InferenceDetailSheet({
  inference,
  isLoading,
  error,
  isOpen,
  onClose,
}: InferenceDetailSheetProps) {
  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="overflow-y-auto w-full sm:max-w-full md:w-5/6">
        <SheetHeader>
          <SheetTitle>
            {inference ? "Inference Details" : "Loading..."}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {isLoading && !inference && (
            <div className="flex items-center justify-center py-12">
              <div className="text-fg-muted text-sm">Loading inference details...</div>
            </div>
          )}

          {error && (
            <div className="flex items-center justify-center py-12">
              <div className="text-destructive text-sm font-medium">{error}</div>
            </div>
          )}

          {inference && !error && (
            <>
              <div className="space-y-2 border-b pb-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-fg-tertiary font-medium">Function:</span>
                    <div className="font-mono">{inference.function_name}</div>
                  </div>
                  <div>
                    <span className="text-fg-tertiary font-medium">Variant:</span>
                    <div className="font-mono">{inference.variant_name}</div>
                  </div>
                  <div>
                    <span className="text-fg-tertiary font-medium">Processing Time:</span>
                    <div>{inference.processing_time_ms}ms</div>
                  </div>
                  <div>
                    <span className="text-fg-tertiary font-medium">Timestamp:</span>
                    <div>{formatDateWithSeconds(new Date(inference.timestamp))}</div>
                  </div>
                </div>
                <div className="text-xs text-fg-tertiary pt-2">
                  <span className="font-medium">ID:</span>{" "}
                  <span className="font-mono">{inference.id}</span>
                </div>
              </div>

              <SectionLayout>
                <SectionHeader heading="Input" />
                <Input
                  messages={inference.input.messages}
                  system={inference.input.system}
                  maxHeight={300}
                />
              </SectionLayout>

              <SectionLayout>
                <SectionHeader heading="Output" />
                <Output output={inference.output} maxHeight={300} />
              </SectionLayout>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
