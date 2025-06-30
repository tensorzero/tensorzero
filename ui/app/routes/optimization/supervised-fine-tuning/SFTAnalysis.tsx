import { AlertTriangleIcon } from "lucide-react";
import {
  BasicInfoItem,
  BasicInfoItemContent,
  BasicInfoItemTitle,
  BasicInfoLayout,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
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

const StatisticItem: React.FC<{
  label: string;
  value: number;
}> = ({ label, value }) => (
  <BasicInfoItem>
    <BasicInfoItemTitle>{label}</BasicInfoItemTitle>
    <BasicInfoItemContent>
      <Chip label={value.toString()} font="mono" />
    </BasicInfoItemContent>
  </BasicInfoItem>
);

export function SFTAnalysis({ status }: { status: SFTJobStatus }) {
  if (status.status === "idle") return null;

  const analysisData = status.analysisData;
  if (!analysisData) return null;

  return (
    <>
      <section className="flex flex-row gap-8">
        <BasicInfoLayout>
          <h4>Examples</h4>
          <StatisticItem label="Total" value={analysisData.numExamples} />
          <StatisticItem
            label="Missing system messages"
            value={analysisData.missingSystemCount}
          />
          <StatisticItem
            label="Missing user messages"
            value={analysisData.missingUserCount}
          />
          <StatisticItem label="Truncated" value={analysisData.tooLongCount} />
        </BasicInfoLayout>

        <BasicInfoLayout>
          <h4>Messages per example</h4>
          <StatisticItem label="Min" value={analysisData.messageCounts.min} />
          <StatisticItem label="Max" value={analysisData.messageCounts.max} />
          <StatisticItem label="Mean" value={analysisData.messageCounts.mean} />
          <StatisticItem
            label="Median"
            value={analysisData.messageCounts.median}
          />
          <StatisticItem label="p5" value={analysisData.messageCounts.p5} />
          <StatisticItem label="p95" value={analysisData.messageCounts.p95} />
        </BasicInfoLayout>
      </section>

      {/* TODO Reusable component for this? */}
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
    </>
  );
}
