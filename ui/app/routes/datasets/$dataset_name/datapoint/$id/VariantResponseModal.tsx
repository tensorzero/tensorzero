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
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import { useFetcher } from "react-router";
import type { JsonInferenceOutput } from "~/utils/clickhouse/common";
import type { ContentBlockOutput } from "~/utils/clickhouse/common";
import { OutputContent } from "~/components/inference/Output";
import type { InferenceResponse } from "~/utils/tensorzero";
import { Card, CardContent } from "~/components/ui/card";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";

interface VariantResponseModalProps {
  isOpen: boolean;
  isLoading: boolean;
  setIsLoading: (isLoading: boolean) => void;
  onClose: () => void;
  datapoint: ParsedDatasetRow;
  selectedVariant: string;
}

interface VariantResponseInfo {
  output?: JsonInferenceOutput | ContentBlockOutput[];
  usage?: InferenceUsage;
}

export function VariantResponseModal({
  isOpen,
  isLoading,
  setIsLoading,
  onClose,
  datapoint,
  selectedVariant,
}: VariantResponseModalProps) {
  const [variantResponse, setVariantResponse] =
    useState<VariantResponseInfo | null>(null);
  const [rawResponse, setRawResponse] = useState<InferenceResponse | null>(
    null,
  );
  const [showRawResponse, setShowRawResponse] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const baselineResponse: VariantResponseInfo = {
    output: datapoint.output,
  };

  const variantInferenceFetcher = useFetcher();

  useEffect(() => {
    if (isOpen) {
      setVariantResponse(null);
      setShowRawResponse(false);
      setError(null);
      const request = {
        function_name: datapoint.function_name,
        input: datapoint.input,
        variant_name: selectedVariant,
        dryrun: true,
      };
      variantInferenceFetcher.submit(
        { data: JSON.stringify(request) },
        {
          method: "POST",
          action: "/api/tensorzero/inference",
        },
      );
      setIsLoading(true);
    }
  }, [isOpen, selectedVariant]);

  useEffect(() => {
    if (
      variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading"
    ) {
      setIsLoading(true);
      return;
    }

    setIsLoading(false);

    if (variantInferenceFetcher.data) {
      setError(null);
      try {
        const inferenceOutput =
          variantInferenceFetcher.data as InferenceResponse;
        const variantResponse: VariantResponseInfo = {
          output:
            "content" in inferenceOutput
              ? inferenceOutput.content
              : inferenceOutput.output,
          usage: inferenceOutput.usage,
        };
        setVariantResponse(variantResponse);
        setRawResponse(inferenceOutput);
      } catch (err) {
        setError("Failed to process response data");
        console.error("Error processing response:", err);
      }
    } else if (
      variantInferenceFetcher.state === "idle" &&
      variantInferenceFetcher.data === undefined
    ) {
      setError("Failed to fetch response. Please try again.");
    }
  }, [variantInferenceFetcher.data, variantInferenceFetcher.state]);

  const ResponseColumn = ({
    title,
    response,
  }: {
    title: string;
    response: VariantResponseInfo | null;
  }) => (
    <div className="flex flex-1 flex-col">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      {response && (
        <>
          <div className="flex-1">
            <h4 className="mb-1 text-xs font-semibold">Output</h4>
            {response.output && (
              <Card>
                <CardContent className="pt-8">
                  <OutputContent output={response.output} />
                </CardContent>
              </Card>
            )}
          </div>
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
      )}
    </div>
  );

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>
            Comparing datapoint vs.{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono">
              {selectedVariant}
            </code>
          </DialogTitle>
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
              <div className="flex min-h-[300px] space-x-4">
                <ResponseColumn title="Original" response={baselineResponse} />
                <ResponseColumn title="New" response={variantResponse} />
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
                  <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-gray-100 p-4 text-xs">
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
