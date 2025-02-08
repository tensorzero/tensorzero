"use client";

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
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import { useFetcher } from "react-router";
import type { JsonInferenceOutput } from "~/utils/clickhouse/common";
import type { ContentBlockOutput } from "~/utils/clickhouse/common";
import Output from "./Output";
import type { InferenceResponse } from "~/utils/tensorzero";

interface VariantResponseModalProps {
  isOpen: boolean;
  isLoading: boolean;
  setIsLoading: (isLoading: boolean) => void;
  onClose: () => void;
  inference: ParsedInferenceRow;
  inferenceUsage: InferenceUsage;
}

interface VariantResponseInfo {
  output: JsonInferenceOutput | ContentBlockOutput[];
  usage?: InferenceUsage;
}

export function VariantResponseModal({
  isOpen,
  isLoading,
  setIsLoading,
  onClose,
  inference,
  inferenceUsage,
}: VariantResponseModalProps) {
  const [variantResponse, setVariantResponse] =
    useState<VariantResponseInfo | null>(null);
  const [showRawResponse, setShowRawResponse] = useState(false);
  const baselineResponse: VariantResponseInfo = {
    output: inference.output,
    usage: inferenceUsage,
  };
  const variant = inference.variant_name;

  const variantInferenceFetcher = useFetcher();

  useEffect(() => {
    if (isOpen) {
      setVariantResponse(null);
      setShowRawResponse(false);
      const request = {
        function_name: inference.function_name,
        input: inference.input,
        variant_name: variant,
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
  }, [isOpen, variant]);

  useEffect(() => {
    if (variantInferenceFetcher.data) {
      setIsLoading(false);
      console.log("variantInferenceFetcher.data", variantInferenceFetcher.data);
      const inferenceOutput = variantInferenceFetcher.data as InferenceResponse;
      console.log("inferenceOutput", inferenceOutput);

      const variantResponse: VariantResponseInfo = {
        output:
          "content" in inferenceOutput
            ? inferenceOutput.content
            : inferenceOutput.output,
        usage: inferenceOutput.usage,
      };
      setVariantResponse(variantResponse);
    }
  }, [variantInferenceFetcher.data]);

  const ResponseColumn = ({
    title,
    response,
  }: {
    title: string;
    response: VariantResponseInfo | null;
  }) => (
    <div className="flex-1">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      {response && (
        <>
          <div className="mb-4">
            <Output output={response.output} />
          </div>
          {response.usage && (
            <div>
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
  console.log("isOpen", isOpen);
  console.log("isLoading", isLoading);
  console.log("variantResponse", variantResponse);
  console.log("baselineResponse", baselineResponse);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>
            Comparing{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm">
              baseline
            </code>{" "}
            vs{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm">
              {variant}
            </code>
          </DialogTitle>
        </DialogHeader>
        <div className="mt-4 max-h-[70vh] overflow-y-auto">
          {isLoading ? (
            <div className="flex h-32 items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : (
            <>
              <div className="flex space-x-4">
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
                    <code>{JSON.stringify(variantResponse, null, 2)}</code>
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
