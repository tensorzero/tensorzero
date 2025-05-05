import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Separator } from "~/components/ui/separator";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import NewOutput from "~/components/inference/NewOutput";
import type { InferenceResponse } from "~/utils/tensorzero";
import { Card, CardContent } from "~/components/ui/card";
import type { VariantResponseInfo } from "~/routes/api/tensorzero/inference.utils";

interface VariantResponseModalProps {
  isOpen: boolean;
  isLoading: boolean;
  onClose: () => void;
  // Use a union type to accept either inference or datapoint
  item: ParsedInferenceRow | ParsedDatasetRow;
  // Make inferenceUsage optional since datasets don't have it by default
  inferenceUsage?: InferenceUsage;
  selectedVariant: string;
  // Add a source property to determine what type of item we're dealing with
  source: "inference" | "datapoint";
  error?: string | null;
  variantResponse: VariantResponseInfo | null;
  rawResponse: InferenceResponse | null;
}

export function VariantResponseModal({
  isOpen,
  isLoading,
  onClose,
  item,
  inferenceUsage,
  selectedVariant,
  source,
  error,
  variantResponse,
  rawResponse,
}: VariantResponseModalProps) {
  const [showRawResponse, setShowRawResponse] = useState(false);

  // Set up baseline response based on source type
  const baselineResponse: VariantResponseInfo = {
    output: item.output,
    usage: source === "inference" ? inferenceUsage : undefined,
  };

  // Get original variant name if available (only for inferences)
  const originalVariant =
    source === "inference"
      ? (item as ParsedInferenceRow).variant_name
      : undefined;

  useEffect(() => {
    // reset when modal opens or closes
    setShowRawResponse(false);
  }, [isOpen]);

  const ResponseColumn = ({
    title,
    response,
    errorMessage,
  }: {
    title: string;
    response: VariantResponseInfo | null;
    errorMessage?: string | null;
  }) => (
    <div className="flex flex-1 flex-col">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      {errorMessage ? (
        <div className="flex-1">
          <Card>
            <CardContent className="pt-8">
              <div className="text-red-600">
                <p className="font-semibold">Error</p>
                <p>{errorMessage}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        response && (
          <>
            {response.output && (
              <div className="flex-1">
                <NewOutput
                  output={response.output}
                  outputSchema={
                    "output_schema" in item ? item.output_schema : undefined
                  }
                />
              </div>
            )}

            {response.usage && (
              <div className="mt-4">
                <h4 className="mb-1 text-xs font-semibold">Usage</h4>
                <p className="text-xs">
                  Input tokens: {response.usage.input_tokens}
                </p>
                <p className="text-xs">
                  Output tokens: {response.usage.output_tokens}
                </p>
              </div>
            )}
          </>
        )
      )}
    </div>
  );

  // Create a dynamic title based on the source
  const getTitle = () => {
    if (source === "inference") {
      return (
        <>
          Comparing{" "}
          <code className="bg-muted rounded px-1.5 py-0.5 font-mono">
            {originalVariant}
          </code>{" "}
          vs.{" "}
          <code className="bg-muted rounded px-1.5 py-0.5 font-mono">
            {selectedVariant}
          </code>
        </>
      );
    } else {
      return (
        <>
          Comparing datapoint vs.{" "}
          <code className="bg-muted rounded px-1.5 py-0.5 font-mono">
            {selectedVariant}
          </code>
        </>
      );
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[90vw]">
        <DialogHeader>
          <DialogTitle>{getTitle()}</DialogTitle>
        </DialogHeader>
        <div className="mt-4 max-h-[70vh] overflow-y-auto">
          {isLoading ? (
            <div className="flex h-32 items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : error ? (
            <div className="flex h-32 items-center justify-center">
              <div className="text-center text-red-600">
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          ) : (
            <>
              <div className="flex flex-col gap-4 md:grid md:min-h-[300px] md:grid-cols-2">
                <ResponseColumn title="Original" response={baselineResponse} />
                <ResponseColumn
                  title="New"
                  response={variantResponse}
                  errorMessage={error}
                />
              </div>
              <Separator className="my-4" />
              <div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowRawResponse(!showRawResponse)}
                  className="w-full justify-between"
                >
                  Raw Response
                  {showRawResponse ? (
                    <ChevronUp className="ml-2 h-4 w-4" />
                  ) : (
                    <ChevronDown className="ml-2 h-4 w-4" />
                  )}
                </Button>
                {showRawResponse && (
                  <pre className="mt-2 overflow-x-auto rounded-md bg-gray-100 p-4 text-xs break-words whitespace-pre-wrap">
                    <code>{JSON.stringify(rawResponse, null, 2)}</code>
                  </pre>
                )}
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
