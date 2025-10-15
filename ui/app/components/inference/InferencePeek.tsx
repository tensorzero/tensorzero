import { Sheet, SheetContent, SheetHeader, SheetTitle } from "~/components/ui/sheet";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import InputSnippet from "~/components/inference/InputSnippet";
import { Output } from "~/components/inference/Output";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { TableItemShortUuid } from "~/components/ui/TableItems";

interface InferencePeekProps {
  inference: ParsedInferenceRow | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export function InferencePeek({ inference, isOpen, onOpenChange }: InferencePeekProps) {
  if (!inference) return null;

  return (
    <Sheet open={isOpen} onOpenChange={onOpenChange}>
      <SheetContent className="bg-bg-secondary overflow-y-auto p-0 w-full md:w-5/6">
        <div className="p-6">
          <SheetHeader className="mb-6">
            <SheetTitle className="flex items-center gap-2">
              <span>Inference</span>
              <TableItemShortUuid id={inference.id} />
            </SheetTitle>
          </SheetHeader>

          <div className="space-y-6">
            <SectionLayout>
              <SectionHeader heading="Input" />
              <InputSnippet
                system={inference.input.system}
                messages={inference.input.messages}
                maxHeight={300}
              />
            </SectionLayout>

            <SectionLayout>
              <SectionHeader heading="Output" />
              <Output
                output={
                  inference.function_type === "json"
                    ? { ...inference.output, schema: inference.output_schema }
                    : inference.output
                }
                maxHeight={300}
              />
            </SectionLayout>

            <SectionLayout>
              <SectionHeader heading="Details" />
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium">Function:</span>{" "}
                  <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">
                    {inference.function_name}
                  </code>
                </div>
                <div>
                  <span className="font-medium">Variant:</span>{" "}
                  <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">
                    {inference.variant_name}
                  </code>
                </div>
                <div>
                  <span className="font-medium">Processing Time:</span>{" "}
                  {inference.processing_time_ms}ms
                </div>
                <div>
                  <span className="font-medium">Timestamp:</span>{" "}
                  {new Date(inference.timestamp).toLocaleString()}
                </div>
              </div>
            </SectionLayout>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}