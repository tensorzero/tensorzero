import { AlertTriangleIcon } from "lucide-react";
import { FirstExample } from "./FirstExample";
import { type SFTJobStatus } from "~/utils/supervised_fine_tuning/common";

interface Message {
  role: string;
  content: string;
}

export interface AnalysisData {
  firstExample?: Message[];
  numExamples: number;
  missingSystemCount: number;
  missingUserCount: number;
  messageCounts: {
    min: number;
    max: number;
    mean: number;
    median: number;
    p5: number;
    p95: number;
  };
  tokenCounts: {
    min: number;
    max: number;
    mean: number;
    median: number;
    p5: number;
    p95: number;
  };
  assistantTokenCounts: {
    min: number;
    max: number;
    mean: number;
    median: number;
    p5: number;
    p95: number;
  };
  tooLongCount: number;
  tokenLimit: number;
}

interface Props {
  status: SFTJobStatus;
}

export function SFTAnalysis({ status }: Props) {
  if (status.status === "idle") return null;

  const analysisData = status.analysisData;
  if (!analysisData) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">Dataset Analysis</h3>

      {analysisData.firstExample && (
        <FirstExample messages={analysisData.firstExample} />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="bg-muted rounded-lg p-4">
          <h4 className="mb-2 font-medium">Basic Statistics</h4>
          <ul className="space-y-1">
            <li className="">
              <code className="text-sm">
                Total examples: {analysisData.numExamples}
              </code>
            </li>
            <li>
              <code className="text-sm">
                Missing system messages: {analysisData.missingSystemCount}
              </code>
            </li>
            <li>
              <code className="text-sm">
                Missing user messages: {analysisData.missingUserCount}
              </code>
            </li>
          </ul>
        </div>

        <div className="bg-muted rounded-lg p-4">
          <h4 className="mb-2 font-medium">Messages per Example</h4>
          <ul className="space-y-1">
            <li>
              <code className="text-sm">
                Min: {analysisData.messageCounts.min} / Max:{" "}
                {analysisData.messageCounts.max}
              </code>
            </li>
            <li>
              <code className="text-sm">
                Mean: {analysisData.messageCounts.mean.toFixed(2)} / Median:{" "}
                {analysisData.messageCounts.median}
              </code>
            </li>
            <li>
              <code className="text-sm">
                p5: {analysisData.messageCounts.p5} / p95:{" "}
                {analysisData.messageCounts.p95}
              </code>
            </li>
          </ul>
        </div>

        <div className="bg-muted rounded-lg p-4">
          <h4 className="mb-2 font-medium">Total Tokens per Example</h4>
          <ul className="space-y-1">
            <li>
              <code className="text-sm">
                Min: {analysisData.tokenCounts.min} / Max:{" "}
                {analysisData.tokenCounts.max}
              </code>
            </li>
            <li>
              <code className="text-sm">
                Mean: {analysisData.tokenCounts.mean.toFixed(2)} / Median:{" "}
                {analysisData.tokenCounts.median}
              </code>
            </li>
            <li>
              <code className="text-sm">
                p5: {analysisData.tokenCounts.p5} / p95:{" "}
                {analysisData.tokenCounts.p95}
              </code>
            </li>
          </ul>
        </div>

        <div className="bg-muted rounded-lg p-4">
          <h4 className="mb-2 font-medium">Assistant Tokens per Example</h4>
          <ul className="space-y-1">
            <li>
              <code className="text-sm">
                Min: {analysisData.assistantTokenCounts.min} / Max:{" "}
                {analysisData.assistantTokenCounts.max}
              </code>
            </li>
            <li>
              <code className="text-sm">
                Mean: {analysisData.assistantTokenCounts.mean.toFixed(2)} /
                Median: {analysisData.assistantTokenCounts.median}
              </code>
            </li>
            <li>
              <code className="text-sm">
                p5: {analysisData.assistantTokenCounts.p5} / p95:{" "}
                {analysisData.assistantTokenCounts.p95}
              </code>
            </li>
          </ul>
        </div>
      </div>

      {analysisData.tooLongCount > 0 && (
        <div className="rounded border-l-4 border-yellow-400 bg-yellow-50 p-4">
          <div className="flex">
            <div className="shrink-0">
              <AlertTriangleIcon className="size-5 text-yellow-400" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                <code className="text-sm">{analysisData.tooLongCount}</code>{" "}
                examples may be over the{" "}
                <code className="text-sm">{analysisData.tokenLimit}</code> token
                limit. These will be truncated during fine-tuning.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
