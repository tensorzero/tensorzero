import { useEffect, useMemo, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import {
  Loader2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Copy,
  Check,
} from "lucide-react";
import { InputIcon, Output, Cost } from "~/components/icons/Icons";
import { Button } from "~/components/ui/button";
import { Separator } from "~/components/ui/separator";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
  StoredInference,
} from "~/types/tensorzero";
import { Card, CardContent } from "~/components/ui/card";
import type { VariantResponseInfo } from "~/routes/api/tensorzero/inference.utils";
import { formatCost } from "~/utils/cost";
import { Link } from "react-router";
import { toInferenceUrl } from "~/utils/urls";
import type { Datapoint, InferenceResponse } from "~/types/tensorzero";
import { CodeEditor } from "~/components/ui/code-editor";
import { diffWords } from "diff";

/** Extract plain text from a chat output for diffing */
function extractTextFromOutput(
  response: VariantResponseInfo | null,
): string | null {
  if (!response?.output) return null;
  if (response.type === "json") {
    const jsonOutput = response.output as JsonInferenceOutput;
    if (jsonOutput?.raw) return jsonOutput.raw;
    if (jsonOutput?.parsed) {
      try {
        return JSON.stringify(jsonOutput.parsed, null, 2);
      } catch {
        return null;
      }
    }
    return null;
  }
  // Chat output: concatenate text blocks
  const blocks = response.output as ContentBlockChatOutput[];
  const textParts = blocks
    .filter((b): b is { type: "text" } & { text: string } => b.type === "text")
    .map((b) => b.text);
  return textParts.length > 0 ? textParts.join("\n") : null;
}

/** Render inline diff between two strings */
function TextDiff({ oldText, newText }: { oldText: string; newText: string }) {
  const parts = useMemo(() => diffWords(oldText, newText), [oldText, newText]);

  return (
    <div className="bg-bg-secondary border-border overflow-auto rounded-lg border p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (part.added) {
          return (
            <span
              key={i}
              className="rounded-xs bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300"
            >
              {part.value}
            </span>
          );
        }
        if (part.removed) {
          return (
            <span
              key={i}
              className="rounded-xs bg-red-100 text-red-800 line-through dark:bg-red-900/40 dark:text-red-300"
            >
              {part.value}
            </span>
          );
        }
        return (
          <span key={i} className="text-fg-secondary">
            {part.value}
          </span>
        );
      })}
    </div>
  );
}

function LoadingState({ selectedVariant }: { selectedVariant: string }) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());

  useEffect(() => {
    startRef.current = Date.now();
    setElapsed(0);
    const interval = setInterval(() => {
      setElapsed(((Date.now() - startRef.current) / 1000) | 0);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  const formatElapsed = (s: number) => {
    if (s < 1) return "0.0s";
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}m ${s % 60}s`;
  };

  return (
    <div className="flex h-48 flex-col items-center justify-center gap-4">
      <Loader2 className="text-fg-tertiary h-8 w-8 animate-spin" />
      <div className="text-center">
        <p className="text-fg-primary text-sm font-medium">
          Running inference with{" "}
          <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs">
            {selectedVariant}
          </code>
        </p>
        <p className="text-fg-muted mt-1 font-mono text-xs tabular-nums">
          {formatElapsed(elapsed)}
        </p>
      </div>
    </div>
  );
}

interface ResponseColumnProps {
  title: string;
  response: VariantResponseInfo | null;
  errorMessage?: string | null;
  inferenceId?: string | null;
  onClose?: () => void;
  actions?: React.ReactNode;
  refreshButton?: React.ReactNode;
}

function CopyTextButton({ text }: { text: string | null }) {
  const [copied, setCopied] = useState(false);
  if (!text) return null;
  return (
    <Button
      variant="ghost"
      size="iconSm"
      className="h-6 w-6 cursor-pointer opacity-50 transition-opacity hover:opacity-100"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      title="Copy text"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </Button>
  );
}

function ResponseColumn({
  title,
  response,
  errorMessage,
  inferenceId,
  onClose,
  actions,
  refreshButton,
}: ResponseColumnProps) {
  const text = extractTextFromOutput(response);
  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {refreshButton}
      <div className="mb-2 flex items-center gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <CopyTextButton text={text} />
      </div>
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
                {response.type === "json" ? (
                  <JsonOutputElement output={response.output} />
                ) : (
                  <ChatOutputElement output={response.output} />
                )}
              </div>
            )}

            {inferenceId && (
              <div className="mt-2 text-xs">
                Inference ID:{" "}
                <Link
                  to={toInferenceUrl(inferenceId)}
                  className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
                  onClick={onClose}
                >
                  {inferenceId}
                </Link>
              </div>
            )}

            <div className="mt-4">{actions}</div>
          </>
        )
      )}
    </div>
  );
}

interface VariantResponseModalProps {
  isOpen: boolean;
  isLoading: boolean;
  onClose: () => void;
  // Use a union type to accept either inference or datapoint
  item: StoredInference | Datapoint;
  // Make inferenceUsage optional since datasets don't have it by default
  inferenceUsage?: InferenceUsage;
  selectedVariant: string;
  // Add a source property to determine what type of item we're dealing with
  source: "inference" | "datapoint";
  error?: string | null;
  variantResponse: VariantResponseInfo | null;
  rawResponse: InferenceResponse | null;
  children?: React.ReactNode;
  onRefresh?: (() => void) | null;
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
  children,
  onRefresh,
}: VariantResponseModalProps) {
  const [showRawResponse, setShowRawResponse] = useState(false);

  // Set up baseline response based on source type
  // Datapoint has `type`, ParsedInferenceRow has `function_type`
  const itemType =
    "type" in item ? item.type : (item as ParsedInferenceRow).function_type;
  const baselineResponse: VariantResponseInfo =
    itemType === "json"
      ? {
          type: "json",
          output: item.output as JsonInferenceOutput | undefined,
          usage: source === "inference" ? inferenceUsage : undefined,
        }
      : {
          type: "chat",
          output: item.output as ContentBlockChatOutput[] | undefined,
          usage: source === "inference" ? inferenceUsage : undefined,
        };

  // Get original variant name if available (only for inferences)
  const originalVariant =
    source === "inference" ? (item as StoredInference).variant_name : undefined;

  const refreshButton = onRefresh && (
    <Button
      aria-label="Refresh variant response"
      variant="ghost"
      size="iconSm"
      className="absolute top-1 right-1 z-5 h-6 w-6 cursor-pointer text-xs opacity-25 transition-opacity hover:opacity-100"
      onClick={onRefresh}
      disabled={isLoading}
    >
      {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw />}
    </Button>
  );

  useEffect(() => {
    // reset when modal opens or closes
    setShowRawResponse(false);
  }, [isOpen]);

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
      <DialogContent
        className="max-h-[90vh] sm:max-w-[90vw]"
        aria-describedby={undefined}
      >
        <DialogHeader>
          <DialogTitle>{getTitle()}</DialogTitle>
        </DialogHeader>
        <div>
          {isLoading ? (
            <LoadingState selectedVariant={selectedVariant} />
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
                  inferenceId={rawResponse?.inference_id}
                  onClose={onClose}
                  refreshButton={refreshButton}
                  actions={children}
                />
              </div>

              <UsageComparison
                baselineUsage={baselineResponse.usage}
                newUsage={variantResponse?.usage}
              />

              <DiffSection
                baselineResponse={baselineResponse}
                variantResponse={variantResponse}
              />

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
                {showRawResponse && rawResponse && (
                  <div className="mt-2">
                    <RawResponseViewer rawResponse={rawResponse} />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DiffSection({
  baselineResponse,
  variantResponse,
}: {
  baselineResponse: VariantResponseInfo;
  variantResponse: VariantResponseInfo | null;
}) {
  const oldText = extractTextFromOutput(baselineResponse);
  const newText = extractTextFromOutput(variantResponse);
  const [showDiff, setShowDiff] = useState(false);

  if (!oldText || !newText || oldText === newText) return null;

  return (
    <>
      <Separator className="my-4" />
      <div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowDiff(!showDiff)}
          className="w-full justify-between"
        >
          Text Diff
          {showDiff ? (
            <ChevronUp className="ml-2 h-4 w-4" />
          ) : (
            <ChevronDown className="ml-2 h-4 w-4" />
          )}
        </Button>
        {showDiff && (
          <div className="mt-2">
            <TextDiff oldText={oldText} newText={newText} />
          </div>
        )}
      </div>
    </>
  );
}

function RawResponseViewer({
  rawResponse,
}: {
  rawResponse: InferenceResponse;
}) {
  const jsonString = useMemo(
    () => JSON.stringify(rawResponse, null, 2),
    [rawResponse],
  );
  return (
    <CodeEditor
      value={jsonString}
      allowedLanguages={["json"]}
      readOnly
      maxHeight="400px"
    />
  );
}

function formatTokenDelta(
  original: number | null | undefined,
  updated: number | null | undefined,
): React.ReactNode {
  if (original == null || updated == null) return null;
  const delta = updated - original;
  if (delta === 0) return <span className="text-fg-muted text-xs">=</span>;
  const sign = delta > 0 ? "+" : "";
  const color =
    delta > 0
      ? "text-red-600 dark:text-red-400"
      : "text-green-600 dark:text-green-400";
  return (
    <span className={`text-xs font-medium ${color}`}>
      {sign}
      {delta.toLocaleString()}
    </span>
  );
}

function formatCostDelta(
  original: number | null | undefined,
  updated: number | null | undefined,
): React.ReactNode {
  if (original == null || updated == null) return null;
  const delta = updated - original;
  if (Math.abs(delta) < 0.000001)
    return <span className="text-fg-muted text-xs">=</span>;
  const sign = delta > 0 ? "+" : "";
  const color =
    delta > 0
      ? "text-red-600 dark:text-red-400"
      : "text-green-600 dark:text-green-400";
  return (
    <span className={`text-xs font-medium ${color}`}>
      {sign}
      {formatCost(delta)}
    </span>
  );
}

function UsageComparison({
  baselineUsage,
  newUsage,
}: {
  baselineUsage?: InferenceUsage;
  newUsage?: InferenceUsage;
}) {
  if (!baselineUsage && !newUsage) return null;

  return (
    <>
      <Separator className="my-4" />
      <div className="bg-bg-secondary border-border rounded-lg border p-3">
        <h4 className="text-fg-primary mb-2 text-xs font-semibold">Usage</h4>
        <div className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-x-4 gap-y-1 text-xs">
          <div className="text-fg-muted font-medium" />
          <div className="text-fg-muted text-center font-medium">Original</div>
          <div className="text-fg-muted text-center font-medium">New</div>
          <div className="text-fg-muted text-center font-medium">Delta</div>

          <div className="text-fg-secondary flex items-center gap-1.5">
            <InputIcon className="text-fg-tertiary h-3.5 w-3.5" />
            Input tokens
          </div>
          <div className="text-fg-primary text-center font-mono tabular-nums">
            {baselineUsage?.input_tokens?.toLocaleString() ?? "—"}
          </div>
          <div className="text-fg-primary text-center font-mono tabular-nums">
            {newUsage?.input_tokens?.toLocaleString() ?? "—"}
          </div>
          <div className="text-center font-mono tabular-nums">
            {formatTokenDelta(
              baselineUsage?.input_tokens,
              newUsage?.input_tokens,
            ) ?? "—"}
          </div>

          <div className="text-fg-secondary flex items-center gap-1.5">
            <Output className="text-fg-tertiary h-3.5 w-3.5" />
            Output tokens
          </div>
          <div className="text-fg-primary text-center font-mono tabular-nums">
            {baselineUsage?.output_tokens?.toLocaleString() ?? "—"}
          </div>
          <div className="text-fg-primary text-center font-mono tabular-nums">
            {newUsage?.output_tokens?.toLocaleString() ?? "—"}
          </div>
          <div className="text-center font-mono tabular-nums">
            {formatTokenDelta(
              baselineUsage?.output_tokens,
              newUsage?.output_tokens,
            ) ?? "—"}
          </div>

          {(baselineUsage?.cost != null || newUsage?.cost != null) && (
            <>
              <div className="text-fg-secondary flex items-center gap-1.5">
                <Cost className="text-fg-tertiary h-3.5 w-3.5" />
                Cost
              </div>
              <div className="text-fg-primary text-center font-mono tabular-nums">
                {baselineUsage?.cost != null
                  ? formatCost(baselineUsage.cost)
                  : "—"}
              </div>
              <div className="text-fg-primary text-center font-mono tabular-nums">
                {newUsage?.cost != null ? formatCost(newUsage.cost) : "—"}
              </div>
              <div className="text-center font-mono tabular-nums">
                {formatCostDelta(baselineUsage?.cost, newUsage?.cost) ?? "—"}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
